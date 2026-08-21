import hashlib
import re
from typing import Optional
from lxml import html as lxml_html
from app.core.logging import logger


BOILERPLATE_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "iframe", "svg", "form", "button"}

BOILERPLATE_CLASSES_IDS = re.compile(
    r"(cookie|consent|banner|nav|navbar|footer|header|sidebar|widget|ad-|advertisement|popup|modal|social-share)",
    re.IGNORECASE
)


class ContentCleaner:
    """Cleans raw HTML by stripping boilerplate, navbars, footers, scripts, and ads."""

    @staticmethod
    def clean_html_lxml(raw_html: str) -> str:
        """Parse raw HTML with lxml and remove boilerplate elements."""
        if not raw_html or not raw_html.strip():
            return ""

        try:
            tree = lxml_html.fromstring(raw_html)

            # Remove tags like script, style, nav, footer
            for tag in BOILERPLATE_TAGS:
                for elem in tree.xpath(f"//{tag}"):
                    elem.getparent().remove(elem)

            # Remove elements with boilerplate class/id patterns
            for elem in list(tree.iter()):
                if elem.getparent() is None:
                    continue
                class_name = elem.get("class", "")
                id_name = elem.get("id", "")
                if BOILERPLATE_CLASSES_IDS.search(class_name) or BOILERPLATE_CLASSES_IDS.search(id_name):
                    # Only remove if it's not the main body/content container
                    if elem.tag not in ("body", "main", "article", "html"):
                        elem.getparent().remove(elem)

            return lxml_html.tostring(tree, encoding="unicode")
        except Exception as exc:
            logger.debug(f"lxml cleaning fallback to regex cleaner: {exc}")
            return ContentCleaner.clean_html_regex(raw_html)

    @staticmethod
    def clean_html_regex(raw_html: str) -> str:
        """Regex fallback cleaner for malformed HTML."""
        if not raw_html:
            return ""

        cleaned = raw_html

        # Strip script, style, nav, footer, header tags and content
        cleaned = re.sub(r"<(script|style|nav|footer|header|noscript|iframe)[^>]*>.*?</\1>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        # Strip self-closing or leftover tags
        cleaned = re.sub(r"<(script|style|nav|footer|header|noscript|iframe)[^>]*/?>", "", cleaned, flags=re.IGNORECASE)

        return cleaned

    @classmethod
    def clean(cls, raw_html: str) -> str:
        """Clean raw HTML string."""
        return cls.clean_html_lxml(raw_html)

    @staticmethod
    def extract_plain_text(clean_html: str) -> str:
        """Extract clean, whitespace-normalized plain text from HTML."""
        if not clean_html:
            return ""

        # Remove HTML comments
        text = re.sub(r"<!--.*?-->", "", clean_html, flags=re.DOTALL)
        # Replace block tags with newline
        text = re.sub(r"<(p|h1|h2|h3|h4|h5|h6|li|tr|div|section|article)[^>]*>", "\n", text, flags=re.IGNORECASE)
        # Replace remaining tags
        text = re.sub(r"<[^>]+>", "", text)
        # Decode common HTML entities
        text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
        # Normalize whitespace per paragraph
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        clean_lines = [line for line in lines if line]

        return "\n\n".join(clean_lines)

    @staticmethod
    def calculate_content_hash(content: str) -> str:
        """Calculate SHA-256 content hash of cleaned text."""
        normalized = re.sub(r"\s+", " ", content).strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
