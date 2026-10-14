"""Auxiliary final evaluation: N64 sampled integers plus the native greedy score."""
from para_agent.diagnose.smoothed_scorer import SmoothedKananaAdapter, essay_seed


class AuxiliaryKananaScorer:
    def __init__(self, native_adapter, rubric_keys, policy, *, sampler_factory=SmoothedKananaAdapter):
        self.native_adapter=native_adapter
        self.rubric_keys=list(rubric_keys)
        self.policy=policy
        self.sampler_factory=sampler_factory
        self.sampler=None

    @property
    def backend(self):
        return self.native_adapter.backend

    def score_for_essay(self, text, prompt, *, essay_id):
        if not essay_id:
            raise ValueError('Original essay ID is required for common-random-number scoring')
        greedy,_=self.native_adapter.score(text,prompt,use_cache=False)
        settings=self.policy['kanana']
        if self.sampler is None:
            self.sampler=self.sampler_factory(self.native_adapter._ensure_backend(),
                batch_size=settings['generation_batch_size'],expected_batch_size=1,
                temperature=settings['temperature'],include_expected=False)
        seed=essay_seed(essay_id,settings['master_seed'])
        sampled=self.sampler.score(prompt,text,n=settings['n'],seed=seed)
        values=sampled['summary'][str(settings['n'])]['integer_mean']['rubrics']
        if len(values)!=len(self.rubric_keys) or sampled['seed']!=seed or sampled['n']!=settings['n']:
            raise ValueError('Smoothed scorer returned an inconsistent result')
        return {'scores':dict(zip(self.rubric_keys,values)),'greedy_scores':greedy,
                'sampling':sampled,'method':'smoothed_integer_mean','n':settings['n'],
                'seed':seed,'metric_role':'auxiliary','policy_version':self.policy['version']}

    def close(self):
        self.native_adapter.close()
