import pandas as pd

from ml.data_pipeline import clean_text, load_raw, make_v1, make_v2


def test_clean_text_lowercase():
    assert clean_text("FREE Prize") == "free prize"


def test_clean_text_spaces_tabs_newlines():
    assert clean_text("hello \t there\n\nfriend ") == "hello there friend"


def test_clean_text_only_spaces():
    assert clean_text(" ") == ""


def test_load_raw_with_quotes(tmp_path):
    # a quote inside a message must not merge lines together
    raw = tmp_path / "sms.txt"
    raw.write_text('ham\tHe said "ok\nspam\tWIN "cash" now\nham\tbye\n', encoding="utf-8")
    df = load_raw(raw)
    assert len(df) == 3
    assert list(df["label"]) == ["ham", "spam", "ham"]


def test_v1_keeps_everything():
    df = pd.DataFrame({"label": ["ham", "ham"], "text": ["hi", "hi"]})
    assert len(make_v1(df)) == 2


def test_v2_removes_exact_duplicates():
    df = pd.DataFrame({"label": ["ham", "ham", "spam"], "text": ["hi", "hi", "win now"]})
    assert len(make_v2(df)) == 2


def test_v2_removes_duplicates_that_differ_only_in_case_or_spaces():
    df = pd.DataFrame({"label": ["spam", "spam"], "text": ["WIN now", "win now"]})
    assert len(make_v2(df)) == 1


def test_v2_removes_blank_messages():
    df = pd.DataFrame({"label": ["ham", "ham"], "text": ["", "ok"]})
    result = make_v2(df)
    assert list(result["text"]) == ["ok"]


def test_v2_does_not_change_the_input():
    df = pd.DataFrame({"label": ["ham"], "text": ["Hello"]})
    make_v2(df)
    assert df["text"][0] == "Hello"
