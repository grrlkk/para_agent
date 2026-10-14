"""Blind structured judging; inputs exclude model, condition, scores and labels."""
from typing import Literal
from pydantic import Field
from para_agent.schema import Schema


class ContentAddition(Schema):
    status: Literal['있음','없음','판단불가']
    added_spans: list[str]
    reason: str = Field(min_length=1)


class ComparedAdditions(Schema):
    A: ContentAddition
    B: ContentAddition


class SemanticJudgment(Schema):
    label: Literal['보존','경미한 뉘앙스 변화','의미 변화']
    reason: str = Field(min_length=1)
    content_addition: ContentAddition


class Preference(Schema):
    winner: Literal['A','B','동률']
    reason: str = Field(min_length=1)


class PairwiseJudgment(Schema):
    correction_accuracy: Preference
    style: Preference
    meaning: Preference
    overall: Preference
    content_addition: ComparedAdditions


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
    def validate(result):
        validate_addition(result.content_addition,item['after'])
    return client.request('judge_semantic',semantic_payload(item),SemanticJudgment,context=context,validate=validate)


def validate_addition(value,text,*,original_available=True):
    if not original_available and value.status!='판단불가':
        raise ValueError('Original text unavailable: content addition must be undetermined')
    if value.status=='있음' and not value.added_spans:
        raise ValueError('Content addition requires literal evidence')
    if value.status!='있음' and value.added_spans:
        raise ValueError('Only a positive addition finding may have added spans')
    if any(not s.strip() or s not in text for s in value.added_spans):
        raise ValueError('Content addition evidence must occur literally in the displayed text')


def judge_pairwise(client,item,*,context,reverse=False):
    payload=pairwise_payload(item,reverse=reverse)
    def validate(result):
        validate_pairwise_result(result,payload)
    result=client.request('judge_pairwise',payload,PairwiseJudgment,context=context,validate=validate)
    return canonical_pairwise(result,reverse)


def validate_pairwise_result(result,payload):
    if payload.get('original') is None and result.meaning.winner != '동률':
        raise ValueError('Original text unavailable: meaning preservation cannot be compared')
    for side in ('A','B'):
        validate_addition(getattr(result.content_addition,side),payload[side],
                          original_available=payload.get('original') is not None)


def canonical_pairwise(result,reverse):
    canonical={
        key:{'winner':restore_order(value.winner,reverse),'reason':value.reason}
        for key,value in ((k,getattr(result,k)) for k in ('correction_accuracy','meaning','style','overall'))}
    canonical['content_addition']={side:getattr(result.content_addition,restore_order(side,reverse)).model_dump()
                                   for side in ('A','B')}
    return {'displayed':result.model_dump(),'canonical':canonical}
