"""Pre-flight check:  python -m app.check

Verifies .env, the MySQL connection, the Gemini key/model, and the documents.
"""

import sys

from sqlalchemy import text

from app.config import get_settings


def main() -> int:
    s = get_settings()
    ok = True

    print("1) Gemini API key ...", "set" if s.google_api_key else "MISSING (add GOOGLE_API_KEY to .env)")
    if s.google_api_key:
        try:
            from google import genai

            client = genai.Client(api_key=s.google_api_key)
            names = sorted(
                m.name.removeprefix("models/")
                for m in client.models.list()
                if "generateContent" in (m.supported_actions or [])
            )
            wanted = s.gemini_model
            print(f"   model '{wanted}':", "available" if wanted in names else "NOT FOUND")
            if wanted not in names:
                ok = False
                print("   Models your key can use:", ", ".join(n for n in names if "gemini" in n))
            print("   Generating a test reply ...", end=" ")
            from app.llm import get_chat_model

            print(get_chat_model().invoke("Reply with the single word: ready").text.strip())
        except Exception as exc:
            ok = False
            print("   ERROR:", exc)
    else:
        ok = False

    print("2) MySQL connection ...", end=" ")
    try:
        from app.db.session import get_engine, init_db

        init_db()  # creates the database and tables if missing
        with get_engine().connect() as conn:
            version = conn.execute(text("SELECT VERSION()")).scalar()
        print(f"OK (MySQL {version}, tables ready)")
    except Exception as exc:
        ok = False
        print("FAILED:", exc)

    print("3) Documents in data/raw/:")
    from app.ingestion.manifest import load_manifest

    for d in load_manifest():
        print(f"   [{'x' if d.exists else ' '}] {d.quarter:8} {d.type:25} {d.filename}")
        ok &= d.exists

    print("4) Vector index:", "built" if s.chroma_path.exists() else "not built (run: python -m app.ingestion)")
    print("\nAll checks passed." if ok else "\nSome checks failed - see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
