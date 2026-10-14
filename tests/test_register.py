import pytest

from para_agent.memory.schema import Profile
from para_agent.register import REGISTERS, classify_ending, register_distribution, update_profile_register


@pytest.mark.parametrize('text,expected', [
    ('감사합니다.','하십시오체'), ('정말 좋습니다!','하십시오체'), ('가십시오.','하십시오체'),
    ('괜찮습니까?','하십시오체'), ('함께 갑시다.','하십시오체'),
    ('집에 가요.','해요체'), ('정말 좋아요.','해요체'), ('학교예요.','해요체'),
    ('학교에 갑니다.','하십시오체'), ('나는 간다.','해라체'), ('함께 가자.','해라체'),
    ('밥을 먹어라.','해라체'), ('왜 그러니?','해라체'),
    ('정말 좋아.','해체'), ('너도 그렇게 생각해?','해체'), ('나는 학생이야.','해체'),
    ('친구에게','기타'), ('#@이름#야!','기타'), ('“안녕하세요.”','기타'),
    ('','기타'), ('캐나다','기타'),
])
def test_suffix_classes(text,expected):
    assert classify_ending(text)[0] == expected


def test_distribution_ties_unknowns_and_first_unambiguous_style_target():
    mixed = register_distribution('나는 간다. 너는 가요. 친구에게')
    assert set(mixed['counts']) == set(REGISTERS)
    assert mixed['counts']['해라체'] == mixed['counts']['해요체'] == mixed['counts']['기타'] == 1
    assert mixed['dominant'] == '기타' and sum(mixed['proportions'].values()) == 1
    profile = Profile(prompt='문항')
    assert update_profile_register(profile, [mixed])['after'] == '기타'
    assert update_profile_register(profile, [register_distribution('나는 간다.')])['after'] == '해라체'
    assert update_profile_register(profile, [register_distribution('좋습니다.')])['after'] == '해라체'

