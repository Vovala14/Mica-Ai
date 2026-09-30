from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mica_r1.topic_consistency import TopicConsistency, topic_anchors


def test_anchor_skips_the_matched_tail_and_intervening_adverbs():
    prompt = "I misplaced my keys yesterday, maybe I left them"
    assert topic_anchors(prompt, 2) == ("keys",)
    prompt = "If you want to get to the airport before rush hour, take the"
    assert topic_anchors(prompt, 2) == ("airport",)


def test_correlated_candidate_outranks_disjoint_candidate():
    # The toy index has 100 train sentences. "parcel" and "noon" occur
    # together twice, while "parcel" and "cave" never co-occur.
    index = SimpleNamespace(
        sentences=[None] * 100,
        postings={
            "parcel": [(0, 0), (1, 0), (2, 0)],
            "noon": [(0, 1), (1, 1), (3, 0)],
            "cave": [(4, 0), (5, 0)],
        },
    )
    topic = TopicConsistency(index)
    prompt = "The parcel should arrive by"
    relevant = topic.score(prompt, " noon.", 2)
    unrelated = topic.score(prompt, " the cave.", 2)
    assert relevant.value > 0
    assert relevant.joint_sentences == 2
    assert unrelated.value == 0
    assert unrelated.joint_sentences == 0
