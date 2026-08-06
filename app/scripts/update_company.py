import requests
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://webandcrafts.com"

PAGES = [
    "",
    "/about-us",
    "/services",
    "/industries",
    "/careers",
    "/contact",
]


def extract_text(url: str) -> str:
    print(f"Fetching {url}")

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted elements
    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "footer",
        "header",
        "nav",
    ]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


def main():

    company_text = "# Web and Craft\n\n"

    for page in PAGES:

        url = BASE_URL + page

        try:

            company_text += (
                f"\n\n# {url}\n\n"
            )

            company_text += extract_text(url)

        except Exception as e:

            print(e)

    output = Path(
        "app/knowledge/company.md"
    )

    output.write_text(
        company_text,
        encoding="utf-8",
    )

    print("company.md updated")


if __name__ == "__main__":
    main()