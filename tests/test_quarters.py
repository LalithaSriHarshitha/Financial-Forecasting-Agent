from app.tools.quarters import next_quarter, normalize_quarter, previous_quarter, quarter_sort_key


def test_normalize():
    assert normalize_quarter("q2 fy27") == "Q2 FY27"
    assert normalize_quarter("Q4 FY2026") == "Q4 FY26"


def test_previous_and_next():
    assert previous_quarter("Q1 FY27") == "Q4 FY26"
    assert previous_quarter("Q2 FY27", 4) == "Q2 FY26"
    assert next_quarter("Q2 FY27") == "Q3 FY27"
    assert next_quarter("Q4 FY26") == "Q1 FY27"


def test_sorting():
    labels = ["Q2 FY27", "Q4 FY26", "Q1 FY27"]
    assert sorted(labels, key=quarter_sort_key) == ["Q4 FY26", "Q1 FY27", "Q2 FY27"]
