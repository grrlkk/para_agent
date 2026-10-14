#!/usr/bin/env python
"""Prepare local-only calibration and, after a real Luna run, blinded human sheets."""
import argparse
import json
from pathlib import Path
import sys
import yaml

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'));sys.dont_write_bytecode=True
from para_agent.artifacts import file_hash,write_json
from para_agent.config import load_config
from para_agent.data.loader import load_heldout_candidates
from para_agent.eval.validation_data import controlled_pairs,select_original_pairs,changed_pairs,sample_human_semantics
from para_agent.eval.labeling import write_human_sheet
from para_agent.eval.final_score import load_completed_run,CONDITIONS
from para_agent.eval.stage3_metrics import operational_metrics,aggregate_operational


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--luna-run',type=Path)
    parser.add_argument('--stage5-manifest',type=Path)
    args=parser.parse_args(argv);out=args.output.resolve()
    cfg=load_config(PROJECT/'configs/stage3_luna.yaml')
    protocol=yaml.safe_load((PROJECT/'configs/stage3_protocol.yaml').read_text())
    settings=protocol['human_validation']
    source_hashes={str(p.relative_to(PROJECT)):file_hash(p) for p in [PROJECT/'configs/stage3_protocol.yaml',
        PROJECT/'configs/judge_haiku55.yaml',PROJECT/'configs/judge_sonnet55.yaml',
        PROJECT/'prompts/judge_semantic.md',PROJECT/'prompts/judge_pairwise.md']}
    setup=out/'setup.json'
    if not setup.exists():
        exclusions=json.loads((PROJECT/'data/splits/evaluation_exclusions_stage1_5_20261014T124844.658852Z_fee9ddb861.json').read_text())
        ids=set(exclusions['exclude_essay_ids']);hashes=set(exclusions['exclude_whitespace_ignored_sha256'])
        if args.stage5_manifest:
            future=json.loads(args.stage5_manifest.read_text())
            ids.update(r['essay_id'] for r in future['sampling']['selected'])
        keys=yaml.safe_load((PROJECT/'configs/contracts.yaml').read_text())['rubric_keys']
        candidates,audit,_=load_heldout_candidates(cfg.data.source,keys,min_paras=3,max_paras=6)
        candidates=[c for c in candidates if c.provenance['whitespace_ignored_sha256'] not in hashes]
        pairs,selection=select_original_pairs(candidates,ids,count=settings['pairwise_pairs'],
            min_gap=settings['minimum_human_total_gap'],seed=settings['seed'])
        write_json(out/'private/controlled.json',controlled_pairs())
        write_json(out/'private/original_pairs.json',pairs)
        reserved=sorted({eid for p in pairs for eid in p['essay_ids']})
        compact=sorted({r['whitespace_ignored_sha256'] for p in pairs for r in p['source_records']})
        registry=PROJECT/'data/splits/evaluation_exclusions_stage3.json'
        reservation={'exclude_essay_ids':reserved,'exclude_whitespace_ignored_sha256':compact,
            'reason':'Judge calibration originals; exclude from future stage-5 evaluation',
            'source':str(out/'private/original_pairs.json')}
        if registry.exists():
            old=json.loads(registry.read_text())
            if old['exclude_essay_ids']!=reserved:raise ValueError('An existing stage-3 reservation differs; do not silently resample')
        else:write_json(registry,reservation)
        write_json(setup,{'protocol':protocol,'source_hashes':source_hashes,'selection':selection,
            'source_audit':audit,'excluded_existing_ids':sorted(ids),'reserved_stage5_exclusions':reserved,
            'stage5_manifest':str(args.stage5_manifest) if args.stage5_manifest else None,
            'stage5_status':'existing_sample_excluded' if args.stage5_manifest else 'not_sampled_yet_reserved_calibration_ids',
            'controlled_count':80,'original_pair_count':20,'judge_api_calls':0,
            'frozen_data_hashes':{name:file_hash(out/'private'/name) for name in ('controlled.json','original_pairs.json')}})
    else:
        saved=json.loads(setup.read_text())
        if saved['source_hashes']!=source_hashes:raise ValueError('Validation protocol or judge prompts changed after preparation')
        for name,digest in saved['frozen_data_hashes'].items():
            if file_hash(out/'private'/name)!=digest:raise ValueError('Frozen calibration data changed')
    if args.luna_run:
        luna=changed_pairs(args.luna_run,protocol['development_ids'],expected_model=protocol['agent_model'])
        haiku=[]
        for path in protocol['historical_haiku_runs']:
            haiku+=changed_pairs(PROJECT/path,protocol['development_ids'],expected_model='claude-haiku-5-5')
        selected,sampling=sample_human_semantics(luna,haiku,count=settings['semantic_pairs'],
            luna_target=settings['target_luna_semantic_pairs'],risk_target=settings['target_haiku_risk_candidates'],seed=settings['seed'])
        write_json(out/'private/actual_semantics.json',selected)
        write_json(out/'private/luna_semantics.json',luna)
        write_human_sheet(out/'human',selected)
        (out/'human/GUIDELINES.md').write_bytes((PROJECT/'docs/HUMAN_SEMANTIC_LABELING.md').read_bytes())
        cfg,essays,finals,prov,acceptance,hashes=load_completed_run(args.luna_run)
        baseline_pairs=[];metrics=[]
        for essay in essays:
            original='\n\n'.join(essay.paragraphs)
            baseline_pairs.append({'item_id':essay.essay_id,'prompt':essay.prompt,'original':original,
                'A':finals['whole_once',essay.essay_id]['final_text'],'B':finals['incremental',essay.essay_id]['final_text'],
                'cluster':essay.essay_id})
            for condition in CONDITIONS:
                turns=[json.loads(p.read_text()) for p in sorted((args.luna_run/condition/essay.essay_id).glob('turn_*.json'))]
                metrics.append({'essay_id':essay.essay_id,'condition':condition,'agent_model':cfg.llm.agent.model,
                    **operational_metrics(turns,original,finals[condition,essay.essay_id]['Y'])})
        write_json(out/'private/luna_pairs.json',baseline_pairs)
        write_json(out/'luna_local_metrics.json',{'source_run':str(args.luna_run.resolve()),'rows':metrics,'conditions':aggregate_operational(metrics),
            'kanana':'pending_separate_local_evaluation','meaning_preservation':'pending_validated_judge',
            'baseline_contains_haiku':False,'api_cost':acceptance.get('estimated_billed_usd')})
        write_json(out/'actual_sample.json',{'sampling':sampling,'luna_run':str(args.luna_run.resolve()),
            'all_luna_changed_pairs':len(luna),'human_pairs':len(selected),'human_labels':'pending',
            'private_sha256':file_hash(out/'private/actual_semantics.json'),
            'blind_csv_sha256':file_hash(out/'human/semantic_labels.csv')})
    print(json.dumps({'output':str(out),'static_prepared':True,'human_sheet_prepared':bool(args.luna_run),'actual_api_calls':0},ensure_ascii=False))


if __name__=='__main__':main()
