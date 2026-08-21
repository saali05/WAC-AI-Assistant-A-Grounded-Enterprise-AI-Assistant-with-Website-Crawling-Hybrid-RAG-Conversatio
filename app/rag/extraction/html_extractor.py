import re
from dataclasses import dataclass, field
from typing import Optional
from lxml import html as lxml_html
from app.rag.extraction.content_cleaner import ContentCleaner


@dataclass
class HeadingSection:
    heading_level: int  # 1 for H1, 2 for H2, 3 for H3, 0 for root/intro
    heading_title: str
    heading_path: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class ExtractedHTML:
    title: str = ""
    description: str = ""
    canonical_url: Optional[str] = None
    main_text: str = ""
    sections: list[HeadingSection] = field(default_factory=list)


class HTMLExtractor:
    """Extracts structured titles, meta tags, and H1/H2/H3 paragraph hierarchy from cleaned HTML."""

    @staticmethod
    def extract_meta(raw_html: str) -> tuple[str, str, Optional[str]]:
        """Extract title, meta description, and canonical link."""
        title = ""
        description = ""
        canonical = None

        if not raw_html:
            return title, description, canonical

        try:
            tree = lxml_html.fromstring(raw_html)

            # Title
            title_nodes = tree.xpath("//title/text()")
            if title_nodes:
                title = title_nodes[0].strip()

            # Description
            desc_nodes = tree.xpath("//meta[translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='description']/@content")
            if desc_nodes:
                description = desc_nodes[0].strip()

            # Canonical
            canon_nodes = tree.xpath("//link[translate(@rel, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='canonical']/@href")
            if canon_nodes:
                canonical = canon_nodes[0].strip()
        except Exception:
            # Fallback regex matching
            title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
            if title_match:
                title = re.sub(r"\s+", " ", title_match.group(1)).strip()

            desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']', raw_html, re.IGNORECASE)
            if desc_match:
                description = desc_match.group(1).strip()

            canon_match = re.search(r'<link[^>]*rel=["\']canonical["\'][^>]*href=["\'](.*?)["\']', raw_html, re.IGNORECASE)
            if canon_match:
                canonical = canon_match.group(1).strip()

        return title, description, canonical

    @classmethod
    def extract(cls, raw_html: str, url: str) -> ExtractedHTML:
        """Extract meta information and hierarchical heading sections from HTML."""
        title, description, canonical = cls.extract_meta(raw_html)
        clean_html = ContentCleaner.clean(raw_html)
        plain_text = ContentCleaner.extract_plain_text(clean_html)

        sections: list[HeadingSection] = []
        current_path: list[str] = []

        try:
            tree = lxml_html.fromstring(clean_html)
            # Find all headings, paragraphs, lists, tables
            nodes = tree.xpath("//h1 | //h2 | //h3 | //p | //ul | //ol | //table")

            current_section = HeadingSection(
                heading_level=0,
                heading_title=title or "Overview",
                heading_path=[]
            )

            for node in nodes:
                tag = node.tag.lower()
                text = ContentCleaner.extract_plain_text(lxml_html.tostring(node, encoding="unicode")).strip()

                if not text:
                    continue

                if tag in ("h1", "h2", "h3"):
                    level = int(tag[1])
                    heading_text = text

                    # Update heading path hierarchy
                    if level == 1:
                        current_path = [heading_text]
                    elif level == 2:
                        current_path = (current_path[:1] if len(current_path) >= 1 else []) + [heading_text]
                    elif level == 3:
                        current_path = (current_path[:2] if len(current_path) >= 2 else current_path) + [heading_text]

                    if current_section.paragraphs:
                        sections.append(current_section)

                    current_section = HeadingSection(
                        heading_level=level,
                        heading_title=heading_text,
                        heading_path=list(current_path)
                    )
                else:
                    current_section.paragraphs.append(text)

            if current_section.paragraphs or current_section.heading_title:
                sections.append(current_section)

        except Exception:
            # Fallback if structural parsing fails
            sections = [HeadingSection(heading_level=0, heading_title=title or "Main Content", heading_path=[], paragraphs=[plain_text])]

        return ExtractedHTML(
            title=title,
            description=description,
            canonical_url=canonical or url,
            main_text=plain_text,
            sections=sections
        )
