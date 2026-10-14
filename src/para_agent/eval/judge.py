"""Blind structured judging; inputs exclude model, condition, scores and labels."""
from typing import Literal
from pydantic import Field
from para_agent.schema import Schema


class SemanticJudgment(Schema):
    label: Literal['보존','경미한 뉘앙스 변화','의미 변화']
    reason: str = Field(min_length=1)


class Preference(Schema):
    winner: Literal['A','B','동률']
    reason: str = Field(min_length=1)


class PairwiseJudgment(Schema):
    style: Preference
    coherence: Preference
    meaning: Preference
    overall: Preference


def semantic_payload(item):
    return {k:item[k] for k in ('prompt','before','after','context_before','context_after')}


def pairwise_payload(item,*,reverse=False):
    a,b=item['A'],item['B']
    return {'prompt':item['prompt'],'original':item.get('original'),
            'A':b if reverse else a,'B':a if reverse else b,
            'task':'원문 글 두 편의 품질 비교' if item.get('original') is None else '동일 원문의 수정본 비교'}


def restore_order(winner,reverse):
    return {'A':'B','B':'A','동률':'동률'}[winner] if reverse else winner


def judge_semantic(client,item,*,context):
    return client.request('judge_semantic',semantic_payload(item),SemanticJudgment,context=context)


def judge_pairwise(client,item,*,context,reverse=False):
    result=client.request('judge_pairwise',pairwise_payload(item,reverse=reverse),PairwiseJudgment,context=context)
    return {'displayed':result.model_dump(),'canonical':{
        key:{'winner':restore_order(value.winner,reverse),'reason':value.reason}
        for key,value in ((k,getattr(result,k)) for k in ('style','coherence','meaning','overall'))}}
