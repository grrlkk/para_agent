#!/usr/bin/env python
"""Read saved stage-1.5 results; compute preregistered comparisons without inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
from para_agent.artifacts import file_hash, write_json
from para_agent.diagnose.smoothed_scorer import primary_decision, summarize_samples, essay_seed
from para_agent.eval.smoothing import (agreement, convergence_summary, paired_stratified_spearman_interval,
                                      block_statistics, paired_block_statistics)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(run):
    protocol = read(run / "preregistration.json")
    assert file_hash(run / "preregistration.json") == read(run / "preparation.json")["preregistration_sha256"]
    assert protocol['primary_n'] == 32 and protocol['primary_method'] == 'integer_mean'
    assert all(file_hash(Path(path)) == digest for path, digest in read(run / "protected_sources.json").items())
    if (run/'analysis_sources.json').exists():
        assert all(file_hash(Path(path))==digest for path,digest in read(run/'analysis_sources.json').items())
    assert read(run/'cohort_revalidation.json')['ok']
    assert all(file_hash(run/path)==digest for path,digest in read(run/'preserved_completed_results.json').items())
    sampling_settings = yaml.safe_load((run/'sampling_settings.yaml').read_text())
    assert sampling_settings['seed']['scheme']=='sha256_essay_id_v1'
    workers = [read(run / f"worker_gpu{g}/acceptance.json") for g in (0,1)]
    assert all(w['ok'] and w['completed'] == w['assigned'] for w in workers)
    cases, aliases = read(run / "cases.json"), read(run / "aliases.json")
    key_info = read(run / "rubric_contract.json")
    keys, names = key_info['rubric_keys'], key_info['rubric_names_ko']
    results = {p.stem: read(p) for g in (0,1) for p in (run / f"worker_gpu{g}/cases").glob('*.json')}
    assert set(results) == {c['case_id'] for c in cases}
    for c in cases:
        r = results[c['case_id']]
        assert r['text_sha256'] == c['text_sha256'] == file_hash(run / c['text_path'])
        assert r['prompt_sha256'] == c['prompt_sha256'] == hashlib.sha256(c['prompt'].encode()).hexdigest()
        assert r['smoothed']['n'] == 64 and len(r['smoothed']['samples']) == 64
        assert r['smoothed']['expected_batch_size'] == 1
        assert r['smoothed']['seed'] == c['seed'] == essay_seed(c['essay_id'],sampling_settings['seed']['master_seed'])
        assert summarize_samples(r['smoothed']['samples']) == r['smoothed']['summary']
    output = run / "analysis"
    output.mkdir(exist_ok=False)
    method_names = [('greedy_expected', 1)] + [(method, n) for method in ('integer_mean','expected_mean') for n in (8,16,32,64)]
    blocks = {(cid,method,n):block_statistics(r['smoothed']['samples'],method,n)
              for cid,r in results.items() for method,n in method_names if method!='greedy_expected'}
    write_json(output/'score_block_statistics.json',[
        {'case_id':cid,'essay_id':results[cid]['essay_id'],'method':method,**row}
        for (cid,method,n),row in blocks.items()])

    def values(r, method, n):
        if method == 'greedy_expected':
            return np.asarray([r['greedy_scores'][k] for k in keys])
        return np.asarray(r['smoothed']['summary'][str(n)][method]['rubrics'])

    def method_key(method, n):
        return method if method == 'greedy_expected' else f'{method}_n{n}'

    indexed_aliases = {(a['essay_id'], a['label']): a for a in aliases}
    rows = []
    for a in aliases:
        r = results[a['case_id']]
        for method, n in method_names:
            vals = values(r, method, n)
            rows.append({'essay_id':a['essay_id'], 'group':a['group'], 'input_variant':a['label'],
                         'case_id':a['case_id'], 'gpu':r['gpu'], 'method':method, 'n':n,
                         **dict(zip(keys, vals.tolist())), 'total':float(vals.sum()),
                         'total_mc_se':None if method=='greedy_expected' else r['smoothed']['summary'][str(n)][method]['total_mc_se'],
                         'total_block_sd':None if method=='greedy_expected' else blocks[a['case_id'],method,n]['total_block_sd'],
                         'block_count':None if method=='greedy_expected' else blocks[a['case_id'],method,n]['block_count'],
                         **{f'human_{k}':a.get('human_scores',{}).get(k) for k in keys},
                         'human_total':sum(a['human_scores'].values()) if 'human_scores' in a else None})
    write_csv(output / 'scores.csv', rows)
    manifest = read(run / 'calibration_manifest.json')
    people = manifest['sampling']['selected']
    ids = [p['essay_id'] for p in people]
    strata = [p['stratum'] for p in people]
    assert len(ids) == len(set(ids)) == 100
    assert not set(ids) & set(manifest['excluded_pilot_ids'])
    human = np.asarray([[indexed_aliases[eid,'canonical']['human_scores'][k] for k in keys] for eid in ids])
    correlations, correlation_rows, matrices, intervals = {}, [], {}, {}
    for representation in ('canonical','raw'):
        correlations[representation] = {}
        for method,n in method_names:
            key = method_key(method,n)
            predictions = np.asarray([values(results[indexed_aliases[eid,representation]['case_id']],method,n) for eid in ids])
            matrices[representation,key] = predictions
            stats = agreement(human,predictions,keys)
            correlations[representation][key] = stats
            correlation_rows.extend({'representation':representation,'method':method,'samples':n,'rubric':k,**row}
                                    for k,row in stats.items())
        for method in ('integer_mean','expected_mean'):
            key = method_key(method,32)
            intervals[f'{representation}/{key}'] = paired_stratified_spearman_interval(
                human.sum(axis=1), matrices[representation,'greedy_expected'].sum(axis=1),
                matrices[representation,key].sum(axis=1),strata)
    write_json(output / 'correlations.json', correlations)
    write_csv(output / 'correlations.csv', correlation_rows)
    write_json(output / 'correlation_difference_intervals.json', intervals)

    comparisons, paired_blocks = [], []
    for a in aliases:
        if a['group'] != 'development' or a['label']=='original_joined':
            continue
        family = ('raw' if a['label']=='original_raw' else 'lf' if a['label']=='c_single_newline'
                  else 'contraction' if a['label']=='apply_whole_once_p1_s5_5'
                  else 'single_edit' if a['label'].startswith('apply_') else 'law_factor')
        base_alias = indexed_aliases[a['essay_id'],'original_joined']
        base, altered = results[base_alias['case_id']], results[a['case_id']]
        assert base['gpu'] == altered['gpu'], 'Do not confound paired variants with device placement'
        assert base['smoothed']['seed']==altered['smoothed']['seed']
        for method,n in method_names:
            b, y = values(base,method,n), values(altered,method,n)
            se, paired = None, None
            if method != 'greedy_expected':
                field = 'integers' if method=='integer_mean' else 'expected'
                diffs = np.asarray([np.sum(yy[field])-np.sum(xx[field]) for xx,yy in
                                    zip(base['smoothed']['samples'][:n],altered['smoothed']['samples'][:n])])
                se = float(diffs.std(ddof=1)/np.sqrt(n))
                paired=paired_block_statistics(base['smoothed']['samples'],altered['smoothed']['samples'],method,n)
                paired_blocks.append({'essay_id':a['essay_id'],'family':family,'variant':a['label'],
                                      'method':method,**paired})
            comparisons.append({'essay_id':a['essay_id'],'family':family,'variant':a['label'],
                                'method':method,'n':n,'baseline_total':float(b.sum()),'variant_total':float(y.sum()),
                                'delta_total':float(y.sum()-b.sum()),'paired_mc_se_diagnostic':se,
                                'baseline_total_block_sd':None if paired is None else blocks[base_alias['case_id'],method,n]['total_block_sd'],
                                'variant_total_block_sd':None if paired is None else blocks[a['case_id'],method,n]['total_block_sd'],
                                'paired_delta_total_block_sd':None if paired is None else paired['total_block_sd'],
                                'block_count':None if paired is None else paired['block_count'],
                                'accepted_sample_attempt_indices_aligned':None if paired is None else paired['accepted_sample_attempt_indices_aligned'],
                                **{f'paired_delta_block_sd_{k}':None if paired is None or paired['rubric_block_sd'] is None else paired['rubric_block_sd'][i] for i,k in enumerate(keys)},
                                **{f'delta_{k}':float(v) for k,v in zip(keys,y-b)}})
    write_csv(output / 'invariance.csv', comparisons)
    write_json(output / 'invariance.json', comparisons)
    write_json(output / 'paired_difference_block_statistics.json', paired_blocks)

    def decision_for(method,n):
        lf = [r['delta_total'] for r in comparisons if r['family']=='lf' and r['method']==method and r['n']==n]
        contractions = [r['delta_total'] for r in comparisons if r['family']=='contraction' and r['method']==method and r['n']==n]
        assert len(lf)==5 and len(contractions)==1
        return primary_decision(lf,contractions,correlations['canonical']['greedy_expected']['total']['spearman'],
                                correlations['canonical'][method_key(method,n)]['total']['spearman'])
    decision = decision_for('integer_mean',32)
    write_json(output / 'decision.json', decision)
    primary_pairs=[r for r in paired_blocks if r['method']=='integer_mean' and r['n']==32 and r['family'] in ('lf','contraction')]
    lf_by_block=[max(abs(r['total_block_means'][b]) for r in primary_pairs if r['family']=='lf') for b in (0,1)]
    contraction_by_block=[max(abs(r['total_block_means'][b]) for r in primary_pairs if r['family']=='contraction') for b in (0,1)]
    rho_by_block=[float(spearmanr(human.sum(axis=1),[
        blocks[indexed_aliases[eid,'canonical']['case_id'],'integer_mean',32]['total_block_means'][b]
        for eid in ids]).statistic) for b in (0,1)]
    mc_decision={'method':'integer_mean','n':32,'block_count':2,'primary_estimate_block':1,
                 'criterion_unchanged':True,'uncertainty_does_not_change_pass_fail':True,
                 'note':'SD of two disjoint 32-draw estimates is imprecise; it is not a confidence interval. Pair matching accepted sample block indices; common seed does not guarantee variance reduction.',
                 'lf_max_abs_delta_by_block':lf_by_block,'lf_max_abs_delta_block_sd':float(np.std(lf_by_block,ddof=1)),
                 'contraction_max_abs_delta_by_block':contraction_by_block,'contraction_max_abs_delta_block_sd':float(np.std(contraction_by_block,ddof=1)),
                 'canonical_total_spearman_by_block':rho_by_block,'canonical_total_spearman_block_sd':float(np.std(rho_by_block,ddof=1)),
                 'primary_comparisons':[{k:r[k] for k in ('essay_id','family','variant','delta_total','baseline_total_block_sd','variant_total_block_sd','paired_delta_total_block_sd','paired_mc_se_diagnostic','block_count')} for r in comparisons if r['method']=='integer_mean' and r['n']==32 and r['family'] in ('lf','contraction')]}
    assert np.isclose(lf_by_block[0],decision['lf_max_abs_total_delta'])
    assert np.isclose(contraction_by_block[0],decision['contraction_max_abs_total_delta'])
    assert np.isclose(rho_by_block[0],decision['smoothed_total_spearman'])
    write_json(output/'decision_monte_carlo.json',mc_decision)
    secondary_gates = []
    for method in ('integer_mean','expected_mean'):
        for n in (8,16,32,64):
            d = decision_for(method,n)
            d.update(method=method,n=n,primary=(method=='integer_mean' and n==32),
                     role='primary_rule' if method=='integer_mean' and n==32 else 'secondary_diagnostic_not_used_to_select_method')
            secondary_gates.append(d)
    write_json(output / 'all_n_gate_diagnostics.json', secondary_gates)

    cohorts = {'development_unique': [results[c['case_id']] for c in cases if c['group']=='development'],
               'calibration_canonical':[results[indexed_aliases[eid,'canonical']['case_id']] for eid in ids],
               'calibration_raw':[results[indexed_aliases[eid,'raw']['case_id']] for eid in ids]}
    convergence = [{'cohort':cohort,**convergence_summary(rs,method,n)} for cohort,rs in cohorts.items()
                   for method in ('integer_mean','expected_mean') for n in (8,16,32,64)]
    write_csv(output / 'convergence.csv', convergence)
    write_json(output / 'convergence.json', convergence)

    attempts = sum(r['smoothed']['attempted'] for r in results.values())
    rejected = sum(len(r['smoothed']['rejected']) for r in results.values())
    accepted = sum(len(r['smoothed']['samples']) for r in results.values())
    assert accepted == len(cases)*64 and attempts == accepted+rejected
    sample_audit = {'unique_inputs':len(cases),'accepted_lines':accepted,'attempted_lines':attempts,
                    'rejected_lines':rejected,'rejection_rate':rejected/attempts,
                    'deterministic_expected_cache_hits':sum(r['smoothed']['deterministic_expected_cache_hits'] for r in results.values()),
                    'unique_teacher_forced_lines':sum(r['smoothed']['deterministic_expected_unique_lines'] for r in results.values()),
                    'main_native_greedy_calls':sum(len((run/f'worker_gpu{g}/greedy_calls.jsonl').read_text().splitlines()) for g in (0,1)),
                    'main_completed_inputs':len(cases),'preserved_inputs_after_resume':len(read(run/'preserved_completed_results.json')),
                    'benchmark_native_greedy_calls':2,
                    'benchmark_sampled_lines':16,'real_llm_api_calls':0,'external_api_cost_usd':0}
    write_json(output / 'sampling_audit.json', sample_audit)
    build_tables(output, keys, names, rows, comparisons, correlations, convergence, secondary_gates, mc_decision)
    plot_results(output, correlations, convergence, secondary_gates)
    write_json(output / 'acceptance.json', {'ok':True,'stage':'1.5','complete':True,
        'existing_evaluation_settings_changed':False,'preregistration_unchanged':True,
        'protected_sources_unchanged':True,'criterion':decision,'sample_audit':sample_audit,
        'essay_id_seed_policy_unchanged':True,'existing_completed_results_unchanged':True,
        'raw_paragraph_revalidation_passed':True,'block_diagnostics_added':True,
        'scoring_worker_wall_seconds':{str(w['gpu']):w['latency_s'] for w in workers},
        'future_excluded_essay_ids':len(read(run/'future_evaluation_exclusions.json')['exclude_essay_ids'])})
    print(json.dumps({'decision':decision,'sampling':sample_audit,'canonical_correlations':{
        key:row['total'] for key,row in correlations['canonical'].items()}},ensure_ascii=False,indent=2))


def build_tables(output, keys, names, rows, comparisons, correlations, convergence, gates, mc_decision):
    text = ['**1.5단계 전체 수치 표**','',
            '점수는 8개 루브릭 합계다. Δ는 변형−두 줄바꿈 원문이며, 소수 셋째 자리까지 표시했다. '
            '판정은 반올림 전 첫 32개 정수 평균으로 수행한다. 기대값 평균과 다른 N은 별도 진단이다.','',
            '**시드와 묶음 오차 계산 규칙**','',
            '`seed = int.from_bytes(SHA256(UTF8("15042:" + essay_id))[:4], "big") % 2147483647`를 유지했다. '
            '같은 글의 원문·변형·모든 조건의 수정본에는 같은 원래 essay_id를 쓴다. '
            '실행 순서, 문항, 본문, 조건명으로 시드를 바꾸지 않는다. 재현에는 같은 모델·LoRA·수치 연산 설정·배치 크기가 필요하다.','',
            '64개 유효 샘플을 순서대로 8개×8묶음, 16개×4묶음, 32개×2묶음으로 나누고, '
            '각 묶음의 평균 점수 사이 표본 표준편차(ddof=1)를 계산한다. 변형 차이는 같은 번호의 묶음끼리 먼저 빼고 표준편차를 계산한다. '
            '이 SD는 N개 평균의 변동을 나타내므로 묶음 수의 제곱근으로 다시 나누지 않는다. '
            'N=64는 한 묶음이므로 SD를 산출할 수 없다. 오류로 제외한 줄과 생성 길이가 다르면 난수 대응이 약해질 수 있어 분산 감소를 가정하지 않았다.','',
            '**개발용 변형: 기존 greedy와 N=32 두 평균 방식의 변화량**','',
            '| 글 ID | 변형 | greedy Δ | 정수 평균 Δ | 기대값 평균 Δ |','|---|---|---:|---:|---:|']
    groups = sorted({(r['essay_id'],r['variant']) for r in comparisons})
    idx = {(r['essay_id'],r['variant'],r['method'],r['n']):r for r in comparisons}
    for eid,label in groups:
        vals=[idx[eid,label,m,n]['delta_total'] for m,n in [('greedy_expected',1),('integer_mean',32),('expected_mean',32)]]
        text.append(f"| {eid} | {label} | " + ' | '.join(f'{v:+.3f}' for v in vals)+' |')
    text += ['', '**판정에 쓰는 N=32 차이와 두 종류의 몬테카를로 오차**','',
             '점수와 차이는 첫 32개 샘플의 평균이다. SD는 32개씩 두 묶음에서 계산한 값이며, 신뢰구간이 아니다. '
             'SD가 크더라도 처음 정한 통과·탈락 기준은 변경하지 않는다.','',
             '| 글 ID | 변형 | 차이 Δ | 원문 점수 SD | 변형 점수 SD | 짝지은 차이 SD |',
             '|---|---|---:|---:|---:|---:|']
    for r in mc_decision['primary_comparisons']:
        text.append(f"| {r['essay_id']} | {r['variant']} | {r['delta_total']:+.3f} | {r['baseline_total_block_sd']:.3f} | {r['variant_total_block_sd']:.3f} | {r['paired_delta_total_block_sd']:.3f} |")
    text += ['', f"LF 최대 절대차의 묶음 간 SD: {mc_decision['lf_max_abs_delta_block_sd']:.3f}; "
             f"축약형 최대 절대차 SD: {mc_decision['contraction_max_abs_delta_block_sd']:.3f}; "
             f"100편 합계 Spearman의 두 묶음 간 SD: {mc_decision['canonical_total_spearman_block_sd']:.5f}.",
             '전체 입력의 루브릭별 점수 SD와 전체 변형의 루브릭별 차이 SD는 score_block_statistics.json, paired_difference_block_statistics.json에 있다.']
    text += ['', '**개발용 원문 5편의 N별 합계**','',
             '| 글 ID | 방식 | N=8 | N=16 | N=32 | N=64 |','|---|---|---:|---:|---:|---:|']
    for eid in sorted({r['essay_id'] for r in rows if r['group']=='development'}):
        for method in ('integer_mean','expected_mean'):
            vals=[next(r['total'] for r in rows if r['essay_id']==eid and r['input_variant']=='original_joined' and r['method']==method and r['n']==n) for n in (8,16,32,64)]
            text.append(f"| {eid} | {method} | "+' | '.join(f'{v:.3f}' for v in vals)+' |')
    for representation in ('canonical','raw'):
        text += ['', f'**100편 {representation} 입력의 루브릭별 상관 — N=32**','',
                 '| 루브릭 | greedy Pearson | greedy Spearman | 정수 Pearson | 정수 Spearman | 기대값 Pearson | 기대값 Spearman |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for key in [*keys,'total']:
            vals=[correlations[representation][m][key][stat]
                  for m in ('greedy_expected','integer_mean_n32','expected_mean_n32') for stat in ('pearson','spearman')]
            text.append(f"| {key} {names.get(key,'합계')} | "+' | '.join(f'{v:.4f}' for v in vals)+' |')
        text += ['', f'**100편 {representation} 입력의 N별 합계 상관**','',
                 '| 방식 | Pearson | Spearman |','|---|---:|---:|']
        for name,stats in correlations[representation].items():
            r=stats['total'];text.append(f"| {name} | {r['pearson']:.5f} | {r['spearman']:.5f} |")
    text += ['', '**100편 canonical 입력의 N별 묶음 간 표준편차**','',
             '각 글의 묶음 SD를 먼저 계산한 뒤 100편에서 요약했다. '
             '기존 첫 N개 내부의 MC SE와 N64 대비 차이는 보조 진단으로 CSV에 함께 보관한다.','',
             '| 방식 | N | 묶음 수 | 평균 점수 SD | 중앙값 점수 SD | 최대 점수 SD | 첫 N개 평균 MC SE |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in convergence:
        if r['cohort']!='calibration_canonical':continue
        sd=lambda key:'산출 불가' if r[key] is None else f'{r[key]:.3f}'
        text.append(f"| {r['method']} | {r['n']} | {r['blocks_per_input']} | {sd('mean_total_block_sd')} | {sd('median_total_block_sd')} | {sd('max_total_block_sd')} | {r['mean_total_mc_se']:.3f} |")
    text += ['', '**판정 조건의 N별 진단**','',
             '주 판정은 integer_mean N=32 한 행뿐이다. 다른 행이 통과해도 사후에 선택해 주 판정을 바꾸지 않는다.','',
             '| 방식 | N | LF 최대 절대차 | 축약형 절대차 | 합계 Spearman | 세 조건 충족 | 주 판정 |',
             '|---|---:|---:|---:|---:|---|---|']
    for r in gates:
        text.append(f"| {r['method']} | {r['n']} | {r['lf_max_abs_total_delta']:.3f} | {r['contraction_max_abs_total_delta']:.3f} | {r['smoothed_total_spearman']:.5f} | {'예' if r['passes'] else '아니오'} | {'예' if r['primary'] else '아니오'} |")
    (output/'tables.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def plot_results(output, correlations, convergence, gates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),constrained_layout=True)
    colors={'integer_mean':'#2166ac','expected_mean':'#b35806'}
    for method,label in [('integer_mean','Sampled integers'),('expected_mean','Conditional expectations')]:
        selected=[r for r in convergence if r['cohort']=='calibration_canonical' and r['method']==method]
        ns=[r['n'] for r in selected]
        axes[0].plot(ns,[np.nan if r['mean_total_block_sd'] is None else r['mean_total_block_sd'] for r in selected],marker='o',label=label,color=colors[method])
        axes[1].plot(ns,[correlations['canonical'][f'{method}_n{n}']['total']['spearman'] for n in ns],marker='o',label=label,color=colors[method])
        selected=[r for r in gates if r['method']==method]
        axes[2].plot(ns,[r['lf_max_abs_total_delta'] for r in selected],marker='o',label=f'{label}: LF max',color=colors[method])
        axes[2].plot(ns,[r['contraction_max_abs_total_delta'] for r in selected],marker='s',linestyle='--',label=f'{label}: contraction',color=colors[method])
    axes[0].set_title('Disjoint-block variability: 100 texts');axes[0].set_ylabel('Mean SD of N-draw total scores')
    axes[1].set_title('Agreement with human total scores');axes[1].set_ylabel('Spearman correlation')
    axes[1].axhline(correlations['canonical']['greedy_expected']['total']['spearman'],color='black',linestyle=':',label='Native greedy')
    axes[2].set_title('Development invariance');axes[2].set_ylabel('Absolute change in rubric sum');axes[2].axhline(1,color='black',linestyle=':',label='Predeclared limit = 1')
    for ax in axes:
        ax.set_xlabel('Number of accepted sampled lines');ax.set_xticks([8,16,32,64]);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(output/'convergence.png',dpi=180)
    fig.savefig(output/'convergence.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run.resolve())
