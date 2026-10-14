#!/usr/bin/env python
"""Cost a frozen judge batch first; --execute requires an explicitly approved budget."""
import argparse
import json
from pathlib import Path
import sys
import yaml

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'));sys.dont_write_bytecode=True
from para_agent.artifacts import write_json,file_hash,json_text
from para_agent.config import ModelConfig,load_config,project_path
from para_agent.llm.anthropic_transport import AnthropicTransport
from para_agent.llm.client import Client
from para_agent.eval.costs import BudgetTransport,token_cost,request_upper_estimate
from para_agent.eval.judge import SemanticJudgment,PairwiseJudgment,semantic_payload,pairwise_payload,judge_semantic,judge_pairwise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--model',choices=['haiku','sonnet'],required=True)
    p.add_argument('--scope',nargs='+',choices=['controlled','original_pairs','actual_semantics','luna_semantics','luna_pairs'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--approved-budget-usd',type=float)
    p.add_argument('--selection',type=Path,help='Passing validation report; required for Luna baseline judging')
    args=p.parse_args(argv)
    config_path=PROJECT/f'configs/judge_{args.model}55.yaml'
    model=ModelConfig.model_validate(yaml.safe_load(config_path.read_text()))
    setup=json.loads((args.data_dir/'setup.json').read_text())
    fixed=setup.get('protocol',{}).get('judge_selection')=='user_fixed_sonnet_two_gates_else_stop'
    if fixed and args.model!='sonnet':raise ValueError('The judge was fixed to Sonnet before validation')
    for name,digest in setup['source_hashes'].items():
        if file_hash(PROJECT/name)!=digest:raise ValueError('Protocol, model settings, or judge prompts changed after freezing')
    if (any(s.startswith('luna_') for s in args.scope) or (fixed and 'actual_semantics' in args.scope)) and args.execute:
        if not args.selection:raise ValueError('A validated selected judge is required before baseline judging')
        decision=json.loads(args.selection.read_text())['selection']
        if decision['status']!='selected' or decision['model']!=model.model:raise ValueError('Judge has not passed required validation gates')
    if len(set(args.scope))!=len(args.scope):raise ValueError('Duplicate scope')
    rates=yaml.safe_load((PROJECT/'configs/api_pricing_2026-10-15.yaml').read_text())['models'][model.model]
    # A supplied SDK stub prevents key loading/network creation during costing.
    adapter=AnthropicTransport(model,Path('.env'),sdk_client=object())
    jobs=[];hashes={}
    for scope in args.scope:
        source=args.data_dir/'private'/f'{scope}.json';hashes[scope]=file_hash(source)
        rows=json.loads(source.read_text())
        if scope in ('controlled','original_pairs') and hashes[scope]!=setup['frozen_data_hashes'][source.name]:
            raise ValueError('Frozen calibration data changed')
        for item in rows:
            pair=scope.endswith('pairs')
            for reverse in ([False,True] if pair else [False]):
                agent='judge_pairwise' if pair else 'judge_semantic'
                schema=PairwiseJudgment if pair else SemanticJudgment
                payload=pairwise_payload(item,reverse=reverse) if pair else semantic_payload(item)
                params=adapter.parameters(schema.model_json_schema(),agent)
                messages=[{'role':'system','content':(PROJECT/f'prompts/{agent}.md').read_text()},
                          {'role':'user','content':json_text(payload)}]
                estimated_input=len(json_text([messages,params]).encode())/2+1024
                estimated_output=640 if pair else 256
                jobs.append({'scope':scope,'item':item,'pair':pair,'reverse':reverse,
                    'expected_cost':token_cost({'input_tokens':estimated_input,'output_tokens':estimated_output},rates),
                    'reservation':request_upper_estimate(messages,params,rates)})
    plan={'model':model.model,'model_config':model.model_dump(),'logical_calls':len(jobs),'max_calls_with_repairs':len(jobs)*3,
          'expected_usd':sum(j['expected_cost'] for j in jobs),
          'all_initial_requests_reserved_max_usd':sum(j['reservation'] for j in jobs),
          'estimate_method':'UTF-8 bytes/2 plus 1024 overhead input; 256 semantic or 640 pairwise output tokens; no cache discount',
          'source_hashes':hashes,'scope':args.scope,'external_api_calls':0}
    if not args.execute:
        write_json(args.output,plan);print(json_text(plan));return
    if not args.approved_budget_usd:raise ValueError('--execute requires an approved positive budget')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'plan.json',plan)
    cfg=load_config(PROJECT/'configs/stage3_luna.yaml')
    raw=AnthropicTransport(model,project_path(cfg.llm.env_file),max_calls=len(jobs)*3)
    transport=BudgetTransport(raw,rates,args.approved_budget_usd)
    client=Client(model=model,transport=transport,prompts=PROJECT/'prompts',
        cache=args.data_dir/'judge_cache'/args.model,call_log=args.output/'llm_calls.jsonl')
    results=[];ok=False
    try:
        for job in jobs:
            item=job['item'];key=item.get('blind_id',item['item_id'])
            context={'item_id':key,'scope':job['scope'],'reverse':job['reverse']}
            result=judge_pairwise(client,item,context=context,reverse=job['reverse']) if job['pair'] else judge_semantic(client,item,context=context).model_dump()
            record={**context,'result':result,'model':model.model}
            results.append(record)
            write_json(args.output/'items'/f"{job['scope']}_{key}_{int(job['reverse'])}.json",record)
        write_json(args.output/'results.json',results);ok=True
    finally:
        transport.close()
        write_json(args.output/'acceptance.json',{'ok':ok,'completed':len(results),'expected':len(jobs),
            'real_api_calls':transport.call_count,'estimated_billed_usd':transport.spent_usd,
            'approved_budget_usd':args.approved_budget_usd,'model':model.model,'source_hashes':hashes})
    print(json_text({'ok':ok,'results':str(args.output),'api_calls':transport.call_count,'usd':transport.spent_usd}))


if __name__=='__main__':main()
