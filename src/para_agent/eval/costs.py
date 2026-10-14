"""Recorded standard-price estimates and a pre-dispatch spending guard."""
import json


def token_cost(usage, rates):
    input_n=usage.get('input_tokens',0)
    output_n=usage.get('output_tokens',0)
    details=usage.get('input_tokens_details') or {}
    cached=details.get('cached_tokens',0) or 0
    writes=details.get('cache_write_tokens',0) or 0
    fresh=max(0,input_n-cached-writes)
    long=input_n>rates['long_context_threshold']
    im=rates['long_input_multiplier'] if long else 1
    om=rates['long_output_multiplier'] if long else 1
    return (im*(fresh*rates['input']+cached*rates['cached_input']+writes*rates['cache_write'])
            +om*output_n*rates['output'])/1e6


def request_upper_estimate(messages,params,rates):
    # UTF-8 bytes upper-bound ordinary byte-BPE text tokens, with additional
    # protocol/schema allowance. Reserve the full output limit before dispatch.
    input_n=len(json.dumps([messages,params],ensure_ascii=False).encode())+4096
    output_n=params.get('max_output_tokens',params.get('max_tokens',2048))
    conservative={**rates,'input':max(rates['input'],rates['cache_write'])}
    return token_cost({'input_tokens':input_n,'output_tokens':output_n},conservative)


class BudgetTransport:
    def __init__(self,transport,rates,budget_usd):
        if budget_usd<=0:raise ValueError('Positive approved budget required')
        self.transport,self.rates,self.budget_usd=transport,rates,budget_usd
        self.spent_usd=0.0

    def __getattr__(self,name):return getattr(self.transport,name)

    def complete(self,*,agent,messages,params):
        self.transport.last_usage={}
        self.transport.last_metadata={}
        reserve=request_upper_estimate(messages,params,self.rates)
        if self.spent_usd+reserve>self.budget_usd:
            raise RuntimeError('Approved budget would be exceeded by the next request reservation')
        try:
            return self.transport.complete(agent=agent,messages=messages,params=params)
        finally:
            self.spent_usd+=token_cost(getattr(self.transport,'last_usage',{}) or {},self.rates)
