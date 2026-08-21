import urllib.robotparser
from urllib.parse import urlparse
import httpx
from app.core.logging import logger
from app.core.config import settings


class RobotsTxtParser:
    """Robots.txt parser, validator, and sitemap extractor with in-memory caching."""

    def __init__(self, user_agent: str = "WAC-AI-Crawler/1.0") -> None:
        self.user_agent = user_agent
        self._parsers: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._sitemaps: dict[str, list[str]] = {}
        self._crawl_delays: dict[str, float | None] = {}

    async def fetch_and_parse(self, base_url: str, client: httpx.AsyncClient | None = None) -> None:
        """Fetch and parse robots.txt for given domain base URL."""
        parsed = urlparse(base_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        if domain in self._parsers:
            return

        robots_url = f"{domain}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)

        sitemaps: list[str] = []
        crawl_delay: float | None = None

        try:
            should_close = False
            if client is None:
                client = httpx.AsyncClient(timeout=settings.RAG_REQUEST_TIMEOUT, follow_redirects=True)
                should_close = True

            response = await client.get(robots_url)
            if response.status_code == 200:
                content = response.text
                rp.parse(content.splitlines())

                # Extract sitemaps and Crawl-delay from robots.txt content
                for line in content.splitlines():
                    clean_line = line.strip()
                    if clean_line.lower().startswith("sitemap:"):
                        sitemap_url = clean_line.split(":", 1)[1].strip()
                        if sitemap_url:
                            sitemaps.append(sitemap_url)
                    elif clean_line.lower().startswith("crawl-delay:"):
                        try:
                            crawl_delay = float(clean_line.split(":", 1)[1].strip())
                        except ValueError:
                            pass

                logger.info(f"Parsed robots.txt from {robots_url}. Discovered {len(sitemaps)} sitemap(s).")
            else:
                # If robots.txt returns 404 or non-200, allow all by default
                rp.parse([])
                logger.info(f"No robots.txt found at {robots_url} (HTTP {response.status_code}). Allowing all URLs.")

            if should_close:
                await client.aclose()

        except Exception as exc:
            logger.warning(f"Could not fetch robots.txt from {robots_url}: {exc}. Defaulting to allow all.")
            rp.parse([])

        self._parsers[domain] = rp
        self._sitemaps[domain] = sitemaps
        self._crawl_delays[domain] = crawl_delay

    def is_allowed(self, url: str) -> bool:
        """Check if URL is allowed according to parsed robots.txt."""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        rp = self._parsers.get(domain)
        if not rp:
            return True  # If robots.txt hasn't been fetched yet, default to True

        return rp.can_fetch(self.user_agent, url) or rp.can_fetch("*", url)

    def get_crawl_delay(self, base_url: str) -> float | None:
        """Get crawl delay for domain if defined in robots.txt."""
        parsed = urlparse(base_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        return self._crawl_delays.get(domain)

    def get_sitemaps(self, base_url: str) -> list[str]:
        """Get list of sitemap URLs extracted from domain's robots.txt."""
        parsed = urlparse(base_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        return self._sitemaps.get(domain, [])
