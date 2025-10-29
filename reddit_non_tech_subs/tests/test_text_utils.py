from reddit_non_tech_subs.text_utils import clean_text


def test_clean_text_removes_urls_and_keeps_text():
    s = "Check this [great post](https://example.com) and also http://foo.bar!"
    out = clean_text(s)
    assert "example.com" not in out
    assert "http://" not in out
    assert "great post" in out


def test_clean_text_normalizes_whitespace():
    s = "Hello\n\nworld\t\t!"
    out = clean_text(s)
    assert out == "Hello world !" or out == "Hello world!" or out.startswith("Hello world")


