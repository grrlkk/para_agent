#!/usr/bin/env python
"""Compare frozen calibration labels and actual human labels; no model calls."""
import argparse
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'));sys.dont_write_bytecode=True
from para_agent.artifacts import write_json,file_hash
from para_agent.eval.agreement import semantic_agreement,pairwise_agreement,select_judge
from para_agent.eval.labeling import read_human_labels


def load_results(paths,model,data_dir):
    index={}
    for path in paths:
        acceptance=json.loads((path/'acceptance.json').read_text())
        if acceptance['ok'] is not True or acceptance['model']!=model:raise ValueError('Need completed results from the requested judge model')
        for scope,digest in acceptance['source_hashes'].items():
            if file_hash(data_dir/'private'/f'{scope}.json')!=digest:raise ValueError('Judge source data changed')
        for row in json.loads((path/'results.json').read_text()):
            key=(row['scope'],row['item_id'],row['reverse'])
            if key in index:raise ValueError('Duplicate judgment; choose one predetermined execution')
            index[key]=row['result']
    return index


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--haiku-results',type=Path,nargs='*',default=[])
    p.add_argument('--sonnet-results',type=Path,nargs='*',default=[])
    p.add_argument('--human-labels',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    controlled=json.loads((args.data_dir/'private/controlled.json').read_text())
    pairs=json.loads((args.data_dir/'private/original_pairs.json').read_text())
    human=None;actual=[]
    if args.human_labels:
        actual=json.loads((args.data_dir/'private/actual_semantics.json').read_text())
        human=read_human_labels(args.human_labels,[r['blind_id'] for r in actual])
    reports={}
    for short,paths in [('haiku',args.haiku_results),('sonnet',args.sonnet_results)]:
        model=f'claude-{short}-5-5';index=load_results(paths,model,args.data_dir);gates={}
        if all(('controlled',r['item_id'],False) in index for r in controlled):
            truth=[r['gold'] for r in controlled]
            predicted=[index['controlled',r['item_id'],False]['label'] for r in controlled]
            gates['controlled_semantic']=semantic_agreement(truth,predicted,[r['cluster'] for r in controlled])
        if all(('original_pairs',r['item_id'],rev) in index for r in pairs for rev in (False,True)):
            truth=[r['gold'] for r in pairs for rev in (False,True)]
            predicted=[index['original_pairs',r['item_id'],rev]['canonical']['overall']['winner'] for r in pairs for rev in (False,True)]
            gates['pairwise']=pairwise_agreement(truth,predicted,[r['cluster'] for r in pairs for rev in (False,True)])
            gates['pairwise']['both_orders_correct_rate']=sum(predicted[2*i]==predicted[2*i+1]==r['gold'] for i,r in enumerate(pairs))/len(pairs)
            gates['pairwise']['order_disagreement_rate']=sum(predicted[2*i]!=predicted[2*i+1] for i in range(len(pairs)))/len(pairs)
            gates['pairwise']['limitation']='Different-original human score ordering does not validate same-essay revision comparison.'
        if human and all(('actual_semantics',r['blind_id'],False) in index for r in actual):
            truth=[human[r['blind_id']] for r in actual]
            predicted=[index['actual_semantics',r['blind_id'],False]['label'] for r in actual]
            gates['human_semantic']=semantic_agreement(truth,predicted,[r['cluster'] for r in actual])
            secondary=semantic_agreement(truth,predicted,[r['blind_id'] for r in actual])
            gates['human_semantic']['secondary_item_bootstrap_ci95']=secondary['ci95']
            gates['human_semantic']['secondary_ci_limitation']='Treats pairs from the same essay as independent; primary CI clusters by essay.'
            gates['human_semantic']['disagreement_ids']=[r['blind_id'] for r,a,b in zip(actual,truth,predicted) if a!=b]
        reports[model]=gates
    result={'models':reports,'selection':select_judge(reports),'human_labels_status':'complete' if human else 'pending',
            'human_labels_sha256':file_hash(args.human_labels) if human else None,
            'threshold_policy':'point estimates: pairwise >=0.7, both binary semantic kappas >=0.6; CI reported separately',
            'ci_limitations':'Actual revision sample has only five essay clusters; controlled variants share eight rule families.'}
    write_json(args.output,result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
