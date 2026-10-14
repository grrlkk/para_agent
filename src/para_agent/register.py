"""Transparent suffix rules for observed Korean sentence-ending distributions."""
from collections import Counter
import re

from para_agent.text import sentence_spans

REGISTERS = ('하십시오체', '해요체', '해라체', '해체', '기타')
RULE_VERSION = 'korean_endings_v1'


def _has_final_bieup(char):
    return bool(char) and 0xAC00 <= ord(char) <= 0xD7A3 and (ord(char)-0xAC00)%28 == 17


def classify_ending(sentence):
    text=sentence.strip()
    quoted=text.rstrip('.!?。！？…')
    if len(quoted)>1 and (quoted[0],quoted[-1]) in [('"','"'),("'","'"),('“','”'),('‘','’'),('「','」'),('『','』')]:
        return '기타','quotation_only'
    stem=text.rstrip('.!?。！？… \t\r\n"\'”’」』)]}')
    if not stem:
        return '기타','empty'
    word=stem.split()[-1]
    if re.search(r'(습니다|습니까|십시오|십시다)$',word):
        return '하십시오체','formal_suffix'
    for ending in ('니다','니까','시다'):
        if word.endswith(ending) and len(word)>len(ending) and _has_final_bieup(word[-len(ending)-1]):
            return '하십시오체','bieup_formal_suffix'
    if re.search(r'(아요|어요|여요|에요|예요|해요|가요|와요|봐요|줘요|돼요|세요|군요|네요|지요|죠|나요|까요|고요|구요|게요|래요|데요|거든요|잖아요)$',word):
        return '해요체','polite_suffix'
    if re.search(r'(아라|어라|여라|해라|하라|너라|거라|으라|느냐|으냐|냐|니|는가|은가|인가|던가)$',word):
        return '해라체','plain_question_or_imperative'
    if word.endswith('다') and word not in {'바다','캐나다','사이다','판다'}:
        return '해라체','plain_declarative'
    if word.endswith('자') and word not in {'과자','의자','상자','남자','여자','모자'}:
        return '해라체','plain_propositive'
    if re.fullmatch(r'(?:#@[^#]+#|친구|엄마|아빠|선생님)[야아]?',stem):
        return '기타','address_or_fragment'
    if word.endswith(('에게','한테','께')):
        return '기타','particle_fragment'
    if re.search(r'(구나|거든|잖아|몰라|아|어|여|해|지|야|네|게|군|래|가|와|봐|줘|돼|써)$',word):
        return '해체','informal_suffix'
    return '기타','unclassified'


def register_distribution(text):
    records=[]
    for item in sentence_spans(text):
        label,rule=classify_ending(item['text'])
        records.append({'sentence_id':item['id'],'text':item['text'],'register':label,'rule':rule})
    counts=Counter(r['register'] for r in records)
    counts={name:counts[name] for name in REGISTERS}
    classified={name:n for name,n in counts.items() if name!='기타' and n}
    winners=[name for name,n in classified.items() if n==max(classified.values())] if classified else []
    return {'rule_version':RULE_VERSION,'sentence_count':len(records),'counts':counts,
            'proportions':{name:n/len(records) if records else 0.0 for name,n in counts.items()},
            'dominant':winners[0] if len(winners)==1 else '기타','sentences':records,
            'method':'Suffix heuristic; quoted-only and unclassified sentences are 기타; tied dominant styles are 기타.'}


def update_profile_register(profile, distributions):
    """Latch the first unambiguous revised-paragraph register as the style target."""
    before=profile.register_
    if before is None:
        profile.register_='기타'
    if profile.register_=='기타':
        for distribution in distributions:
            if distribution['dominant']!='기타':
                profile.register_=distribution['dominant']
                break
    return {'before':before,'after':profile.register_,'source':'code',
            'policy':'first_unambiguous_revised_paragraph; keep an established target',
            'rule_version':RULE_VERSION}
