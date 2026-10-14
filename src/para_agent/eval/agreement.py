"""Predeclared validation gates with cluster-bootstrap confidence intervals."""
from collections import Counter
import math
import numpy as np

BINARY={'보존':'보존','경미한 뉘앙스 변화':'변화','의미 변화':'변화'}


def kappa(truth,predicted):
    if len(truth)!=len(predicted) or not truth:raise ValueError('Paired nonempty labels required')
    n=len(truth);a,b=Counter(truth),Counter(predicted)
    observed=sum(x==y for x,y in zip(truth,predicted))/n
    expected=sum(a[k]*b[k] for k in set(a)|set(b))/(n*n)
    return (observed-expected)/(1-expected) if expected<1 else None


def bootstrap_metric(truth,predicted,clusters,metric,*,seed=315042,resamples=10000):
    if not(len(truth)==len(predicted)==len(clusters)) or not truth:
        raise ValueError('Need aligned labels and clusters')
    grouped={g:[i for i,c in enumerate(clusters) if c==g] for g in sorted(set(clusters))}
    groups=list(grouped);rng=np.random.default_rng(seed);values=[]
    for _ in range(resamples):
        indices=[i for g in rng.choice(groups,size=len(groups),replace=True) for i in grouped[g]]
        value=metric([truth[i] for i in indices],[predicted[i] for i in indices])
        if value is not None and math.isfinite(value):values.append(value)
    return {'estimate':metric(truth,predicted),'ci95':np.quantile(values,[.025,.975]).tolist() if values else None,
            'n':len(truth),'clusters':len(groups),'resamples':resamples,'valid_resamples':len(values),
            'method':'cluster_percentile_bootstrap','seed':seed}


def semantic_agreement(truth,predicted,clusters,**kwargs):
    a,b=[BINARY[x] for x in truth],[BINARY[x] for x in predicted]
    result=bootstrap_metric(a,b,clusters,kappa,**kwargs)
    result['threshold']=0.6
    result['passes']=result['estimate'] is not None and result['estimate']>=0.6
    result['confusion']={label:{p:sum(x==label and y==p for x,y in zip(a,b)) for p in ('보존','변화')} for label in ('보존','변화')}
    result['three_class_agreement']=sum(x==y for x,y in zip(truth,predicted))/len(truth)
    return result


def pairwise_agreement(truth,predicted,clusters,**kwargs):
    result=bootstrap_metric(truth,predicted,clusters,lambda a,b:sum(x==y for x,y in zip(a,b))/len(a),**kwargs)
    result.update(threshold=.7,passes=result['estimate']>=.7,ties_count_as_incorrect=True)
    return result


def select_judge(results):
    for model in ('claude-haiku-5-5','claude-sonnet-5-5'):
        gates=results.get(model,{})
        if all(gates.get(name,{}).get('passes') is True for name in ('pairwise','controlled_semantic','human_semantic')):
            return {'model':model,'status':'selected','rule':'cheapest_passes_all_three'}
        if any(gates.get(name,{}).get('passes') is False for name in ('pairwise','controlled_semantic','human_semantic')):
            continue
        if any(name not in gates for name in ('pairwise','controlled_semantic','human_semantic')):
            return {'model':None,'status':'pending','reason':'Cheaper candidate validation is incomplete'}
    return {'model':None,'status':'stop','reason':'Neither candidate passed all three gates'}
