"""Pre-flight check:  python -m app.check

Verifies the LLM provider (Ollama or Gemini), the MySQL connection, and the documents.
"""

import sys

import requests
from sqlalchemy import text

from app.config import get_settings


def _check_ollama(s) -> bool:
    print(f"1) Ollama at {s.ollama_base_url} ...", end=" ")
    try:
        tags = requests.get(f"{s.ollama_base_url}/api/tags", timeout=5).json()
    except Exception:
        print("NOT RUNNING. Install from https://ollama.com and start the Ollama app.")
        return False
    installed = {m["name"] for m in tags.get("models", [])}
    installed |= {n.removesuffix(":latest") for n in installed}
    print("running")
    ok = True
    for role, name in (("chat", s.chat_model_name), ("tool", s.tool_model_name),
                       ("embedding", s.embedding_model_name)):
        present = name in installed or name.removesuffix(":latest") in installed
        print(f"   {role:9} model '{name}':", "installed" if present else f"MISSING -> run: ollama pull {name}")
        ok &= present
    if ok:
        from app.llm import get_chat_model

        print("   Generating a test reply ...", end=" ", flush=True)
        try:
            print(get_chat_model().invoke("Reply with the single word: ready").text.strip()[:40])
        except Exception as exc:
            print("ERROR:", exc)
            ok = False
    return ok


def _check_gemini(s) -> bool:
    print("1) Gemini API key ...", "set" if s.google_api_key else "MISSING (add GOOGLE_API_KEY to .env)")
    if not s.google_api_key:
        return False
    try:
        from google import genai

        client = genai.Client(api_key=s.google_api_key)
        names = sorted(
            m.name.removeprefix("models/")
            for m in client.models.list()
            if "generateContent" in (m.supported_actions or [])
        )
        ok = True
        for name in {s.chat_model_name, s.tool_model_name}:
            print(f"   model '{name}':", "available" if name in names else "NOT FOUND")
            ok &= name in names
        if not ok:
            print("   Models your key can use:", ", ".join(n for n in names if "gemini" in n))
            return False
        from app.llm import get_chat_model

        print("   Generating a test reply ...", end=" ", flush=True)
        print(get_chat_model().invoke("Reply with the single word: ready").text.strip()[:40])
        return True
    except Exception as exc:
        print("   ERROR:", exc)
        return False


def main() -> int:
    s = get_settings()
    print(f"LLM provider: {s.llm_provider}  (change with LLM_PROVIDER in .env)\n")
    ok = _check_ollama(s) if s.llm_provider == "ollama" else _check_gemini(s)

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

    built = s.chroma_path.exists()
    print(f"4) Vector index ({s.embedding_model_name}):",
          "built" if built else "not built (run: python -m app.ingestion --skip-download)")
    ok &= built
    print("\nAll checks passed." if ok else "\nSome checks failed - see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
