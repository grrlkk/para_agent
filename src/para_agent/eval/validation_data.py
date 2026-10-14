"""Deterministic validation material; labels and provenance stay in private manifests."""
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import random

from para_agent.sentence_diff import sentence_diff
from para_agent.artifacts import file_hash
from .final_score import load_completed_run


def identifier(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:16]


def controlled_pairs():
    rows=[]
    subjects=['민수','지수','영희','철수','유나','준호','수아','지우','서희','도희']
    for name in subjects:
        examples=[
            ('spacing',f'{name}는 책 을 읽는다.',f'{name}는 책을 읽는다.','보존'),
            ('spelling',f'{name}는 책을 읽었읍니다.',f'{name}는 책을 읽었습니다.','보존'),
            ('contraction',f'{name}는 숙제를 하였습니다.',f'{name}는 숙제를 했습니다.','보존'),
            ('particle',f'{name}는 친구을 만났다.',f'{name}는 친구를 만났다.','보존'),
            ('negation',f'{name}는 책을 읽는다.',f'{name}는 책을 읽지 않는다.','의미 변화'),
            ('modality',f'{name}는 경기에서 이길 수 있다.',f'{name}는 경기에서 이기기도 한다.','의미 변화'),
            ('tense',f'{name}는 학교에 간다.',f'{name}는 학교에 갔다.','의미 변화'),
            ('emotion',f'{name}는 친구에게 두려움을 느낀다.',f'{name}는 친구에게 부담을 느낀다.','의미 변화'),
        ]
        for category,before,after,label in examples:
            rows.append({'item_id':identifier([category,before,after]),'prompt':'문장의 뜻을 비교하세요.',
                'before':before,'after':after,'context_before':before,'context_after':after,
                'gold':label,'cluster':category,'category':category,'source':'rule_generated'})
    assert len(rows)==80
    return rows


def select_original_pairs(candidates,excluded_ids,*,count=20,min_gap=16,seed=315042):
    groups=defaultdict(list)
    for c in candidates:
        if c.essay.essay_id not in excluded_ids:groups[c.essay.prompt].append(c)
    eligible=[]
    for prompt,items in sorted(groups.items()):
        items=sorted(items,key=lambda x:(x.total,x.essay.essay_id))
        if len(items)<2:continue
        low,high=items[0],items[-1]
        if high.total-low.total<min_gap:continue
        if any(sum(high.provenance[k])<=sum(low.provenance[k]) for k in ('rater_1','rater_2')):continue
        eligible.append((prompt,low,high))
    if len(eligible)<count:raise ValueError(f'Only {len(eligible)} distinct prompts meet the predeclared gap')
    rng=random.Random(seed);chosen=rng.sample(eligible,count);rows=[]
    for prompt,low,high in chosen:
        a,b=(low,high) if rng.randrange(2)==0 else (high,low)
        rows.append({'item_id':identifier([prompt,a.essay.essay_id,b.essay.essay_id]),
            'prompt':prompt,'original':None,'A':'\n\n'.join(a.essay.paragraphs),'B':'\n\n'.join(b.essay.paragraphs),
            'gold':'A' if a.total>b.total else 'B','cluster':identifier(prompt),
            'essay_ids':[a.essay.essay_id,b.essay.essay_id],
            'human_scores':{'A':a.total,'B':b.total},'gap':high.total-low.total,
            'source_records':[a.provenance,b.provenance]})
    return rows,{'available_prompt_groups':len(eligible),'selected_pairs':len(rows),'min_gap':min_gap,
                 'both_raters_agree_on_direction':True,'unique_prompts':len(rows),'seed':seed}


def changed_pairs(run,development_ids,*,expected_model=None):
    cfg,essays,finals,provenance,acceptance,hashes=load_completed_run(Path(run))
    if set(e.essay_id for e in essays)!=set(development_ids):
        raise ValueError('Only the fixed development five may supply revision pairs')
    if expected_model and cfg.llm.agent.model!=expected_model:raise ValueError('Unexpected source agent model')
    prompts={e.essay_id:e.prompt for e in essays};rows=[]
    for path in sorted(Path(run).glob('*/*/turn_*.json')):
        turn=json.loads(path.read_text());before=turn['x'];after=turn['reviser_output']['revised_paragraph']
        for index,change in enumerate(sentence_diff(before,after)):
            rows.append({'item_id':identifier([str(Path(run).resolve()),turn['condition'],turn['essay_id'],turn['turn'],index]),
                'prompt':prompts[turn['essay_id']],'before':change['before'],'after':change['after'],
                'context_before':before,'context_after':after,'cluster':turn['essay_id'],
                'essay_id':turn['essay_id'],'condition':turn['condition'],'turn':turn['turn'],
                'source_run':str(Path(run).resolve()),'agent_model':cfg.llm.agent.model,
                'change':change,'source_sha256':file_hash(path),
                'pair_hash':identifier([change['before'],change['after']])})
    return rows


def risk_priority(item):
    before,after=item['before'],item['after']
    known=(('두려움' in before and '부담' in after) or ('이길 수 있다' in before and '이기기도 한다' in after))
    signals=['않','못','수 있다','수도','것 같','때문','두려','부담','항상','모두','반드시','했다','한다']
    return 100 if known else sum(before.count(w)!=after.count(w) for w in signals)


def sample_human_semantics(luna,haiku,*,count=50,luna_target=25,risk_target=20,seed=315042):
    if not luna:raise ValueError('Need actual changed Luna pairs; do not substitute mock or Haiku data')
    rng=random.Random(seed);selected=[];seen=set()
    def add(items,n,reason):
        if n<=0:return
        added=0
        for item in items:
            if item['pair_hash'] in seen:continue
            selected.append({**item,'sampling_reason':reason});seen.add(item['pair_hash']);added+=1
            if added==n:break
    by_group=defaultdict(list)
    for item in luna:by_group[item['essay_id'],item['condition']].append(item)
    for pool in by_group.values():rng.shuffle(pool)
    balanced=[]
    while any(by_group.values()):
        for group in sorted(by_group):
            if by_group[group]:balanced.append(by_group[group].pop())
    add(balanced,min(luna_target,count-len(selected)),'luna_balanced_by_essay_and_condition')
    risks=sorted(haiku,key=lambda x:(-risk_priority(x),x['item_id']))
    add([x for x in risks if risk_priority(x)>0],min(risk_target,count-len(selected)),
        'historical_risk_candidate_not_a_human_label')
    pool=list(haiku)+list(luna);rng.shuffle(pool)
    if len(selected)<count:add(pool,count-len(selected),'random_remaining_unique_pair')
    if len(selected)!=count:raise ValueError('Insufficient unique actual revision pairs')
    rng.shuffle(selected)
    for i,row in enumerate(selected):row['blind_id']=f'M{i+1:03d}'
    return selected,{'n':len(selected),'seed':seed,'models':dict(Counter(r['agent_model'] for r in selected)),
                    'risk_candidates':sum(risk_priority(r)>0 for r in selected),
                    'known_failure_candidates':sum(risk_priority(r)==100 for r in selected),
                    'purpose':'Enriched validation set, not an unbiased preservation-rate estimate'}
