from pathlib import Path

COMPANY_PROMPT = Path(
    "app/data/company_profile.md"
).read_text(encoding="utf-8")