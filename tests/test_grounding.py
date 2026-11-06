import importlib

from langchain_core.documents import Document

from app.schemas.financials import QualitativeInsights
from app.tools.grounding import verify_quote

qa = importlib.import_module("app.tools.qualitative_analysis")

DOCS = [
    Document(
        page_content="[Samir Seksaria (management, prepared remarks)]: Looking ahead into the second half, "
        "the acquisition-related dilution, particularly in MHP, is likely to impact by about 50 basis points.",
        metadata={"quarter": "Q2 FY27", "speaker": "Samir Seksaria", "role": "management"},
    ),
    Document(
        page_content="[Ravi Menon (analyst, qa)]: Will the furlough impact be bigger this year?",
        metadata={"quarter": "Q2 FY27", "speaker": "Ravi Menon", "role": "analyst"},
    ),
]


def test_verify_quote_accepts_real_and_rejects_invented():
    ok, src = verify_quote("the acquisition-related dilution, particularly in MHP, is likely to impact by about 50 basis points", DOCS, "Q2 FY27")
    assert ok and src.metadata["speaker"] == "Samir Seksaria"
    assert not verify_quote("We expect revenue to grow 8% next quarter on strong BFSI demand", DOCS)[0]


def test_wrong_quarter_label_is_corrected():
    ok, src = verify_quote("dilution, particularly in MHP, is likely to impact by about 50 basis points", DOCS, "Q1 FY27")
    assert ok and src.metadata["quarter"] == "Q2 FY27"


def test_ground_statements_drops_unsupported_and_fixes_speaker():
    insights = QualitativeInsights(
        sentiment="cautious", sentiment_rationale="r",
        forward_looking_statements=[
            {"statement": "MHP dilution is likely to impact by about 50 basis points", "quarter": "Q2 FY27", "speaker": "CEO"},
            {"statement": "Margins will expand to 28% next quarter", "quarter": "Q2 FY27"},
        ],
    )
    grounded, dropped = qa.ground_statements(insights, DOCS)
    assert dropped == 1
    assert grounded.forward_looking_statements[0].speaker == "Samir Seksaria"
