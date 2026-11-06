"""Speaker-aware parsing and chunking of TCS earnings-call transcripts.

A transcript is split into speaker turns ("K Krithivasan: ..."). Each turn is tagged with
  * role     - management | analyst | moderator (analysts are the people the moderator
               introduces with "from the line of <name>"),
  * section  - prepared_remarks | qa (Q&A starts at the moderator's "question-and-answer" line),
  * page     - page of the PDF where the turn starts.
In the Q&A, a management answer is chunked together with the analyst question it answers,
so a search for a topic finds the answer even when only the question names the topic.
"""

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

NAME = r"[A-Z][A-Za-z.'\-]*(?: [A-Z][A-Za-z.'\-]*){0,3}"
LABEL_RE = re.compile(rf"^\s*({NAME}):(?:\s+(.*)|\s*$)")
FOOTER_RE = re.compile(r"^\s*Page \d+ of \d+\s*$")
ANALYST_RE = re.compile(rf"(?:from the line ?of|question (?:is |comes )?from)\s+({NAME})")
NOT_SPEAKERS = {"Note", "Sub", "Encl", "Date", "Page"}

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
QUESTION_CONTEXT_CHARS = 350


@dataclass
class Turn:
    speaker: str
    text: str
    page: int
    role: str = "management"
    section: str = "prepared_remarks"


def parse_turns(pages: list[str]) -> list[Turn]:
    """Split transcript pages into speaker turns, starting at the moderator's first line."""
    turns: list[Turn] = []
    started = False
    for page_no, text in enumerate(pages, start=1):
        for line in text.splitlines():
            if FOOTER_RE.match(line):
                continue
            m = LABEL_RE.match(line)
            speaker = m.group(1).strip() if m else None
            is_label = bool(speaker) and speaker not in NOT_SPEAKERS and (
                speaker == "Moderator" or " " in speaker
            )
            if is_label:
                started = True
                turns.append(Turn(speaker=speaker, text=(m.group(2) or "").strip(), page=page_no))
            elif started and turns and line.strip():
                turns[-1].text += " " + line.strip()
    for t in turns:
        t.text = re.sub(r"\s+", " ", t.text).strip()
        t.text = re.sub(r"(\w)- (\w)", r"\1-\2", t.text)  # re-join words hyphenated at line ends
    return [t for t in turns if t.text]


def tag_turns(turns: list[Turn]) -> list[Turn]:
    analysts = {re.sub(r"\s+", " ", a) for t in turns if t.speaker == "Moderator"
                for a in ANALYST_RE.findall(t.text)}
    # Fallback: in the Q&A, whoever speaks right after the moderator hands over is an analyst
    # (unless they already spoke in the prepared remarks, i.e. management).
    prepared, qa_started = set(), False
    for prev, t in zip([None, *turns], turns):
        if t.speaker == "Moderator" and ("question-and-answer" in t.text.lower() or "first question" in t.text.lower()):
            qa_started = True
        if not qa_started and t.speaker != "Moderator":
            prepared.add(t.speaker)
        elif qa_started and prev is not None and prev.speaker == "Moderator" and t.speaker != "Moderator":
            if t.speaker not in prepared and "question" in prev.text.lower():
                analysts.add(t.speaker)
    in_qa = False
    for t in turns:
        if t.speaker == "Moderator":
            t.role = "moderator"
            if "question-and-answer" in t.text.lower() or "first question" in t.text.lower():
                in_qa = True
        elif t.speaker in analysts:
            t.role = "analyst"
        t.section = "qa" if in_qa else "prepared_remarks"
    return turns


def chunk_transcript(pages: list[str], doc_id: str, quarter: str, period_end: str, title: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    turns = tag_turns(parse_turns(pages))
    chunks: list[Document] = []
    last_question: Turn | None = None
    for turn_no, t in enumerate(turns):
        if t.role == "moderator":
            continue
        if t.role == "analyst":
            last_question = t
        context = ""
        if t.role == "management" and t.section == "qa" and last_question:
            q = last_question.text[:QUESTION_CONTEXT_CHARS]
            context = f"[Question from {last_question.speaker} (analyst)]: {q}\n"
        for i, piece in enumerate(splitter.split_text(t.text)):
            chunks.append(
                Document(
                    page_content=f"{context}[{t.speaker} ({t.role}, {t.section.replace('_', ' ')})]: {piece}",
                    metadata={
                        "doc_id": doc_id,
                        "quarter": quarter,
                        "period_end": period_end,
                        "page": t.page,
                        "turn": turn_no,
                        "chunk": i,
                        "speaker": t.speaker,
                        "role": t.role,
                        "section": t.section,
                        "source": title,
                    },
                )
            )
    return chunks
