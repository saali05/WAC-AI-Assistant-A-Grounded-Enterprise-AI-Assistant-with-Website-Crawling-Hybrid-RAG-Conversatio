import xml.etree.ElementTree as ET
from datetime import datetime, UTC
from typing import Optional
import httpx
from app.core.config import settings
from app.core.logging import logger
from app.rag.crawler.url_normalizer import URLNormalizer
from app.rag.exceptions import CrawlerException


class SitemapEntry:
    def __init__(self, url: str, lastmod: Optional[datetime] = None) -> None:
        self.url = url
        self.lastmod = lastmod


class SitemapParser:
    """XML Sitemap and Sitemap Index parser for page discovery."""

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        self.client = client

    async def fetch_sitemap(self, sitemap_url: str) -> str:
        """Fetch XML content from sitemap URL."""
        should_close = False
        client = self.client
        if client is None:
            client = httpx.AsyncClient(timeout=settings.RAG_REQUEST_TIMEOUT, follow_redirects=True)
            should_close = True

        try:
            response = await client.get(sitemap_url)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            logger.warning(f"Failed to fetch sitemap from {sitemap_url}: {exc}")
            raise CrawlerException(f"Failed to fetch sitemap from {sitemap_url}: {exc}")
        finally:
            if should_close:
                await client.aclose()

    def parse_xml_content(self, xml_content: str) -> list[str]:
        """Parse XML string and return list of child sitemap URLs or page URLs."""
        urls: list[str] = []
        try:
            root = ET.fromstring(xml_content)
        except Exception as exc:
            logger.warning(f"Failed to parse XML sitemap content: {exc}")
            return []

        # Remove XML namespace prefix if present
        def clean_tag(tag: str) -> str:
            return tag.split("}", 1)[1] if "}" in tag else tag

        # Check if sitemap index or normal sitemap
        root_tag = clean_tag(root.tag)

        if root_tag == "sitemapindex":
            # Extract child sitemap URLs
            for elem in root.findall(".//*"):
                if clean_tag(elem.tag) == "loc" and elem.text:
                    urls.append(elem.text.strip())
        elif root_tag == "urlset":
            # Extract page URLs
            for elem in root.findall(".//*"):
                if clean_tag(elem.tag) == "loc" and elem.text:
                    urls.append(elem.text.strip())

        return urls

    async def discover_urls(self, sitemap_url: str) -> list[SitemapEntry]:
        """
        Discover all page URLs recursively from sitemap URL or sitemap index.
        Returns normalized entries with optional lastmod timestamps.
        """
        discovered_entries: list[SitemapEntry] = []
        visited_sitemaps: set[str] = set()
        sitemaps_to_visit = [sitemap_url]

        while sitemaps_to_visit:
            current_sitemap = sitemaps_to_visit.pop(0)
            if current_sitemap in visited_sitemaps:
                continue
            visited_sitemaps.add(current_sitemap)

            try:
                xml_content = await self.fetch_sitemap(current_sitemap)
                root = ET.fromstring(xml_content)
            except Exception as exc:
                logger.warning(f"Skipping unparseable sitemap {current_sitemap}: {exc}")
                continue

            def clean_tag(tag: str) -> str:
                return tag.split("}", 1)[1] if "}" in tag else tag

            root_tag = clean_tag(root.tag)

            if root_tag == "sitemapindex":
                for sitemap_node in root.findall(".//*"):
                    if clean_tag(sitemap_node.tag) == "loc" and sitemap_node.text:
                        sitemaps_to_visit.append(sitemap_node.text.strip())
            elif root_tag == "urlset":
                # Process each <url> entry
                for url_node in root:
                    if clean_tag(url_node.tag) != "url":
                        continue

                    loc_val: Optional[str] = None
                    lastmod_dt: Optional[datetime] = None

                    for child in url_node:
                        tag_name = clean_tag(child.tag)
                        if tag_name == "loc" and child.text:
                            loc_val = child.text.strip()
                        elif tag_name == "lastmod" and child.text:
                            try:
                                raw_date = child.text.strip()
                                if raw_date.endswith("Z"):
                                    raw_date = raw_date[:-1] + "+00:00"
                                lastmod_dt = datetime.fromisoformat(raw_date)
                            except Exception:
                                pass

                    if loc_val:
                        try:
                            normalized_url = URLNormalizer.normalize(loc_val)
                            discovered_entries.append(SitemapEntry(url=normalized_url, lastmod=lastmod_dt))
                        except Exception:
                            pass

        # Sort entries: pages with lastmod date first (descending)
        discovered_entries.sort(
            key=lambda entry: entry.lastmod or datetime.min.replace(tzinfo=UTC),
            reverse=True
        )

        return discovered_entries
