from reddit_non_tech_subs.sentiment import _compound_to_label, hf_label_to_vader_like


def test_vader_label_mapping():
    assert _compound_to_label(0.2) == "pos"
    assert _compound_to_label(-0.2) == "neg"
    assert _compound_to_label(0.0) == "neu"


def test_hf_label_to_vader_like():
    assert hf_label_to_vader_like("positive") == "pos"
    assert hf_label_to_vader_like("negative") == "neg"
    assert hf_label_to_vader_like("neutral") == "neu"
    assert hf_label_to_vader_like("POS") == "pos"


