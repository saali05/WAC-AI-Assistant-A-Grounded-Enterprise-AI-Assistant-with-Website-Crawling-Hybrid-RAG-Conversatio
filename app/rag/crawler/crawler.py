import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
from app.core.config import settings
from app.core.logging import logger
from app.rag.crawler.rate_limiter import RateLimiter
from app.rag.crawler.robots import RobotsTxtParser
from app.rag.crawler.sitemap import SitemapParser
from app.rag.crawler.url_normalizer import URLNormalizer
from app.rag.exceptions import CrawlerException, SSRFProtectionException


@dataclass
class CrawledPage:
    url: str
    canonical_url: str
    html: str
    http_status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    depth: int = 0
    crawled_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    rendering_used: str = "httpx"  # "httpx" or "playwright"


class WebCrawler:
    """Production async web crawler with robots.txt, sitemaps, rate limiting, and SSRF security."""

    def __init__(
        self,
        allowed_domains: Optional[list[str]] = None,
        max_pages: Optional[int] = None,
        max_depth: Optional[int] = None,
        timeout: Optional[int] = None,
        concurrency: Optional[int] = None,
        crawl_delay: Optional[float] = None,
        user_agent: str = "WAC-AI-Crawler/1.0"
    ) -> None:
        self.allowed_domains = allowed_domains or settings.allowed_domains_list
        self.max_pages = max_pages or settings.RAG_MAX_PAGES
        self.max_depth = max_depth or settings.RAG_MAX_DEPTH
        self.timeout = timeout or settings.RAG_REQUEST_TIMEOUT
        self.user_agent = user_agent

        self.robots_parser = RobotsTxtParser(user_agent=self.user_agent)
        self.rate_limiter = RateLimiter(concurrency=concurrency, crawl_delay=crawl_delay)

        self.visited_urls: Set[str] = set()
        self.failed_urls: dict[str, str] = {}

    async def fetch_page_httpx(
        self,
        client: httpx.AsyncClient,
        url: str,
        retries: int = 2
    ) -> Optional[CrawledPage]:
        """Fetch page content asynchronously using HTTPX with exponential backoff retries."""
        for attempt in range(retries + 1):
            try:
                response = await client.get(
                    url,
                    headers={"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"},
                    timeout=self.timeout,
                    follow_redirects=True
                )

                if response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} for URL: {url}")
                    return None

                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    logger.info(f"Skipping non-HTML content ({content_type}) at {url}")
                    return None

                canonical = str(response.url)
                try:
                    canonical = URLNormalizer.normalize(canonical)
                except Exception:
                    pass

                return CrawledPage(
                    url=url,
                    canonical_url=canonical,
                    html=response.text,
                    http_status=response.status_code,
                    headers=dict(response.headers),
                    rendering_used="httpx"
                )

            except Exception as exc:
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.warning(f"Failed to fetch {url} after {retries + 1} attempts: {exc}")
                    self.failed_urls[url] = str(exc)
                    return None

        return None

    async def fetch_page_playwright_fallback(self, url: str) -> Optional[CrawledPage]:
        """Optional Playwright rendering fallback for client-side JavaScript pages."""
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(user_agent=self.user_agent)
                page = await context.new_page()
                await page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)
                content = await page.content()
                final_url = page.url
                await browser.close()

                try:
                    canonical = URLNormalizer.normalize(final_url)
                except Exception:
                    canonical = final_url

                logger.info(f"Rendered JS page via Playwright: {url}")
                return CrawledPage(
                    url=url,
                    canonical_url=canonical,
                    html=content,
                    http_status=200,
                    rendering_used="playwright"
                )
        except Exception as exc:
            logger.debug(f"Playwright fallback not available or failed for {url}: {exc}")
            return None

    def extract_internal_links(self, html: str, base_url: str) -> list[str]:
        """Extract valid, allowed internal links from HTML source code."""
        links: list[str] = []
        href_pattern = re.compile(r'href=["\'](.*?)["\']', re.IGNORECASE)

        for match in href_pattern.finditer(html):
            raw_href = match.group(1).strip()
            if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
                continue

            try:
                abs_url = urljoin(base_url, raw_href)
                normalized = URLNormalizer.normalize(abs_url)
                if URLNormalizer.validate_domain(normalized, self.allowed_domains):
                    links.append(normalized)
            except Exception:
                pass

        return list(dict.fromkeys(links))  # Remove duplicates preserving order

    async def crawl(self, start_urls: list[str]) -> list[CrawledPage]:
        """
        Main crawling routine:
        1. Discover URLs via Sitemap & robots.txt
        2. Fallback to recursive link crawling
        3. Respect max_pages, max_depth, robots.txt, rate limits & SSRF protection
        """
        crawled_pages: list[CrawledPage] = []

        if not start_urls:
            return crawled_pages

        # Normalize start URLs & inspect robots.txt
        valid_start_urls: list[str] = []
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            for raw_url in start_urls:
                try:
                    norm = URLNormalizer.normalize(raw_url)
                    valid_start_urls.append(norm)
                    await self.robots_parser.fetch_and_parse(norm, client=client)
                except Exception as exc:
                    logger.warning(f"Invalid start URL {raw_url}: {exc}")

        if not valid_start_urls:
            return crawled_pages

        # Discover sitemap entries
        sitemap_parser = SitemapParser()
        sitemap_urls_to_crawl: list[str] = []

        for root_url in valid_start_urls:
            discovered_sitemaps = self.robots_parser.get_sitemaps(root_url)
            if not discovered_sitemaps:
                parsed_root = urlparse(root_url)
                discovered_sitemaps = [f"{parsed_root.scheme}://{parsed_root.netloc}/sitemap.xml"]

            for sm_url in discovered_sitemaps:
                try:
                    entries = await sitemap_parser.discover_urls(sm_url)
                    for entry in entries:
                        sitemap_urls_to_crawl.append(entry.url)
                except Exception as exc:
                    logger.debug(f"Could not parse sitemap {sm_url}: {exc}")

        # Remove duplicate sitemap URLs
        sitemap_urls_to_crawl = list(dict.fromkeys(sitemap_urls_to_crawl))
        logger.info(f"Sitemap discovery completed. Total sitemap URLs found: {len(sitemap_urls_to_crawl)}")

        # Prepare queue: (url, depth)
        queue: list[tuple[str, int]] = []

        for sm_url in sitemap_urls_to_crawl:
            queue.append((sm_url, 0))

        for root_url in valid_start_urls:
            if root_url not in [q[0] for q in queue]:
                queue.append((root_url, 0))

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            while queue and len(crawled_pages) < self.max_pages:
                current_url, depth = queue.pop(0)

                if current_url in self.visited_urls:
                    continue
                self.visited_urls.add(current_url)

                if depth > self.max_depth:
                    continue

                if not self.robots_parser.is_allowed(current_url):
                    logger.info(f"Skipping {current_url}: blocked by robots.txt")
                    continue

                # Rate limiting acquire
                async with self.rate_limiter:
                    page = await self.fetch_page_httpx(client, current_url)

                if page is None or not page.html.strip():
                    # Attempt Playwright fallback if content insufficient
                    page = await self.fetch_page_playwright_fallback(current_url)

                if page is not None and page.html.strip():
                    page.depth = depth
                    crawled_pages.append(page)
                    logger.info(f"Crawled ({len(crawled_pages)}/{self.max_pages}): {current_url} [depth={depth}]")

                    # Extract internal links for deeper discovery
                    if depth < self.max_depth and len(crawled_pages) < self.max_pages:
                        extracted = self.extract_internal_links(page.html, current_url)
                        for link in extracted:
                            if link not in self.visited_urls:
                                queue.append((link, depth + 1))

        logger.info(f"Crawl completed. Total pages crawled: {len(crawled_pages)}")
        return crawled_pages
