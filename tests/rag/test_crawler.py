import asyncio
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx

from app.rag.crawler.crawler import WebCrawler, CrawledPage
from app.rag.crawler.sitemap import SitemapEntry
from app.rag.models import RAGDocumentModel, CrawlRunModel
from app.services.crawl_service import CrawlService


@pytest.mark.anyio
async def test_crawler_link_extraction():
    crawler = WebCrawler(allowed_domains=["webandcrafts.com"])
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <a href="/services">Services</a>
        <a href="https://webandcrafts.com/about?utm_source=test">About Us</a>
        <a href="https://external.com/link">External Link</a>
        <a href="javascript:void(0)">JS Action</a>
    </body>
    </html>
    """
    links = crawler.extract_internal_links(html_content, base_url="https://webandcrafts.com")
    assert len(links) == 2
    assert "https://webandcrafts.com/services" in links
    assert "https://webandcrafts.com/about" in links
    assert "https://external.com/link" not in links


@pytest.mark.anyio
async def test_crawled_page_dataclass():
    page = CrawledPage(
        url="https://webandcrafts.com/services",
        canonical_url="https://webandcrafts.com/services",
        html="<html><body><h1>WAC Services</h1></body></html>",
        http_status=200,
        depth=1
    )
    assert page.http_status == 200
    assert page.depth == 1
    assert "WAC Services" in page.html


def test_is_content_insufficient_heuristics():
    """Verify conservative JS-shell and insufficient content detection."""
    # 1. Empty or whitespace
    assert WebCrawler._is_content_insufficient("") is True
    assert WebCrawler._is_content_insufficient("   \n\t  ") is True

    # 2. React / Vue SPA shell with script
    react_shell = '<html><head><title>App</title></head><body><div id="root"></div><script src="/static/js/main.js"></script></body></html>'
    assert WebCrawler._is_content_insufficient(react_shell) is True

    vue_shell = '<html><body><div id="app"></div><script src="/js/app.js"></script></body></html>'
    assert WebCrawler._is_content_insufficient(vue_shell) is True

    angular_shell = '<html><body><app-root></app-root><script src="/runtime.js"></script></body></html>'
    assert WebCrawler._is_content_insufficient(angular_shell) is True

    # 3. Very low text (< 10 words)
    tiny_html = "<html><body><p>Hello world</p></body></html>"
    assert WebCrawler._is_content_insufficient(tiny_html) is True

    # 4. Rich content with substantial text (> 30 words)
    rich_html = """
    <html>
    <head><title>Webandcrafts Services</title></head>
    <body>
        <h1>Webandcrafts Enterprise Digital Solutions</h1>
        <p>Webandcrafts is a global digital transformation company providing cutting-edge web development,
        mobile applications, artificial intelligence, cloud architecture, and e-commerce platforms.
        Our team of expert engineers crafts scalable and secure enterprise software tailored to modern business needs.</p>
        <p>We leverage modern technologies like React, Node.js, Python, Laravel, AWS, and MongoDB to deliver excellence.</p>
    </body>
    </html>
    """
    assert WebCrawler._is_content_insufficient(rich_html) is False


@pytest.mark.anyio
async def test_url_discovery_start_urls_and_sitemaps():
    """Verify start URLs and sitemap URLs are properly recorded in discovered_urls."""
    crawler = WebCrawler(allowed_domains=["webandcrafts.com"], max_pages=10)

    # Mock robots parser
    crawler.robots_parser.fetch_and_parse = AsyncMock()
    crawler.robots_parser.get_sitemaps = MagicMock(return_value=["https://webandcrafts.com/sitemap.xml"])
    crawler.robots_parser.is_allowed = MagicMock(return_value=True)

    # Mock sitemap parser
    mock_sitemap_entries = [
        SitemapEntry(url="https://webandcrafts.com/services"),
        SitemapEntry(url="https://webandcrafts.com/about"),
        SitemapEntry(url="https://external.com/blog"),  # out-of-domain
    ]

    # Mock page fetching
    async def mock_fetch_httpx(client, url, retries=2):
        return CrawledPage(
            url=url,
            canonical_url=url,
            html="<html><body><h1>Title</h1><p>" + "word " * 40 + "</p></body></html>",
            http_status=200,
        )

    crawler.fetch_page_httpx = AsyncMock(side_effect=mock_fetch_httpx)

    with patch("app.rag.crawler.crawler.SitemapParser.discover_urls", new=AsyncMock(return_value=mock_sitemap_entries)):
        pages = await crawler.crawl(["https://webandcrafts.com/"])

    # 1. Start URL appears in discovered_urls
    assert "https://webandcrafts.com/" in crawler.discovered_urls
    # 2. Sitemap URLs tracked as discovered
    assert "https://webandcrafts.com/services" in crawler.discovered_urls
    assert "https://webandcrafts.com/about" in crawler.discovered_urls
    # 5. Out of domain links are not discovered
    assert "https://external.com/blog" not in crawler.discovered_urls
    # 6. visited_urls remains separate from discovered_urls
    assert len(crawler.visited_urls) > 0
    assert crawler.visited_urls.issubset(crawler.discovered_urls)


@pytest.mark.anyio
async def test_internal_links_discovery_and_deduplication():
    """Verify internal links discovered from pages are added to discovered_urls and deduplicated."""
    crawler = WebCrawler(allowed_domains=["webandcrafts.com"], max_pages=10, max_depth=2)

    crawler.robots_parser.fetch_and_parse = AsyncMock()
    crawler.robots_parser.get_sitemaps = MagicMock(return_value=[])
    crawler.robots_parser.is_allowed = MagicMock(return_value=True)

    # Page 1 contains links to Page 2 and Page 3 (and duplicate link to Page 2)
    page1_html = """
    <html><body>
        <h1>Home</h1>
        <p>""" + "content " * 40 + """</p>
        <a href="https://webandcrafts.com/page-a">Page A</a>
        <a href="https://webandcrafts.com/page-a">Page A Duplicate</a>
        <a href="https://webandcrafts.com/page-b">Page B</a>
        <a href="https://otherdomain.com/page-c">Other Domain</a>
    </body></html>
    """

    page_other_html = "<html><body><h1>Sub</h1><p>" + "content " * 40 + "</p></body></html>"

    async def mock_fetch_httpx(client, url, retries=2):
        if url in ("https://webandcrafts.com", "https://webandcrafts.com/"):
            return CrawledPage(url=url, canonical_url=url, html=page1_html)
        return CrawledPage(url=url, canonical_url=url, html=page_other_html)

    crawler.fetch_page_httpx = AsyncMock(side_effect=mock_fetch_httpx)

    with patch("app.rag.crawler.crawler.SitemapParser.discover_urls", new=AsyncMock(return_value=[])):
        pages = await crawler.crawl(["https://webandcrafts.com/"])

    # Discovered should contain start URL + page-a + page-b (no otherdomain)
    assert "https://webandcrafts.com/" in crawler.discovered_urls
    assert "https://webandcrafts.com/page-a" in crawler.discovered_urls
    assert "https://webandcrafts.com/page-b" in crawler.discovered_urls
    assert "https://otherdomain.com/page-c" not in crawler.discovered_urls

    # Duplicate count test: page-a is in set only once
    matching_page_a = [u for u in crawler.discovered_urls if u == "https://webandcrafts.com/page-a"]
    assert len(matching_page_a) == 1
    assert len(pages) == 3


@pytest.mark.anyio
async def test_js_shell_detection_triggers_playwright_fallback():
    """Verify that an insufficient/JS-shell page triggers Playwright fallback."""
    crawler = WebCrawler(allowed_domains=["webandcrafts.com"], max_pages=2)
    crawler.robots_parser.fetch_and_parse = AsyncMock()
    crawler.robots_parser.get_sitemaps = MagicMock(return_value=[])
    crawler.robots_parser.is_allowed = MagicMock(return_value=True)

    # HTTPX returns an empty React shell
    js_shell_html = '<html><body><div id="root"></div><script src="app.js"></script></body></html>'
    httpx_page = CrawledPage(
        url="https://webandcrafts.com/app",
        canonical_url="https://webandcrafts.com/app",
        html=js_shell_html,
        rendering_used="httpx"
    )

    # Playwright returns fully rendered HTML
    rendered_html = "<html><body><h1>Rendered by Playwright</h1><p>" + "rendered text " * 30 + "</p></body></html>"
    playwright_page = CrawledPage(
        url="https://webandcrafts.com/app",
        canonical_url="https://webandcrafts.com/app",
        html=rendered_html,
        rendering_used="playwright"
    )

    crawler.fetch_page_httpx = AsyncMock(return_value=httpx_page)
    crawler.fetch_page_playwright_fallback = AsyncMock(return_value=playwright_page)

    with patch("app.rag.crawler.crawler.SitemapParser.discover_urls", new=AsyncMock(return_value=[])):
        pages = await crawler.crawl(["https://webandcrafts.com/app"])

    assert len(pages) == 1
    assert pages[0].rendering_used == "playwright"
    assert "Rendered by Playwright" in pages[0].html
    crawler.fetch_page_playwright_fallback.assert_awaited_once_with("https://webandcrafts.com/app")


@pytest.mark.anyio
async def test_playwright_failure_falls_back_to_httpx():
    """Verify that if Playwright fails, crawler retains the HTTPX page rather than losing it."""
    crawler = WebCrawler(allowed_domains=["webandcrafts.com"], max_pages=2)
    crawler.robots_parser.fetch_and_parse = AsyncMock()
    crawler.robots_parser.get_sitemaps = MagicMock(return_value=[])
    crawler.robots_parser.is_allowed = MagicMock(return_value=True)

    httpx_page = CrawledPage(
        url="https://webandcrafts.com/app",
        canonical_url="https://webandcrafts.com/app",
        html='<html><body><div id="root">Partial content here</div><script src="app.js"></script></body></html>',
        rendering_used="httpx"
    )

    crawler.fetch_page_httpx = AsyncMock(return_value=httpx_page)
    # Playwright fails or is not installed
    crawler.fetch_page_playwright_fallback = AsyncMock(return_value=None)

    with patch("app.rag.crawler.crawler.SitemapParser.discover_urls", new=AsyncMock(return_value=[])):
        pages = await crawler.crawl(["https://webandcrafts.com/app"])

    assert len(pages) == 1
    assert pages[0].rendering_used == "httpx"
    assert "Partial content here" in pages[0].html


@pytest.mark.anyio
async def test_max_pages_and_max_depth_enforcement():
    """Verify crawler respects max_pages and max_depth bounds."""
    crawler = WebCrawler(allowed_domains=["webandcrafts.com"], max_pages=2, max_depth=1)
    crawler.robots_parser.fetch_and_parse = AsyncMock()
    crawler.robots_parser.get_sitemaps = MagicMock(return_value=[])
    crawler.robots_parser.is_allowed = MagicMock(return_value=True)

    def generate_html(url):
        return f"""
        <html><body>
            <h1>Page {url}</h1>
            <p>{"words " * 40}</p>
            <a href="https://webandcrafts.com/depth1-a">D1 A</a>
            <a href="https://webandcrafts.com/depth1-b">D1 B</a>
            <a href="https://webandcrafts.com/depth1-c">D1 C</a>
        </body></html>
        """

    async def mock_fetch(client, url, retries=2):
        return CrawledPage(url=url, canonical_url=url, html=generate_html(url))

    crawler.fetch_page_httpx = AsyncMock(side_effect=mock_fetch)

    with patch("app.rag.crawler.crawler.SitemapParser.discover_urls", new=AsyncMock(return_value=[])):
        pages = await crawler.crawl(["https://webandcrafts.com/"])

    # Should stop at max_pages = 2
    assert len(pages) == 2


@pytest.mark.anyio
async def test_robots_txt_blocking():
    """Verify crawler skips URLs blocked by robots.txt."""
    crawler = WebCrawler(allowed_domains=["webandcrafts.com"], max_pages=5)
    crawler.robots_parser.fetch_and_parse = AsyncMock()
    crawler.robots_parser.get_sitemaps = MagicMock(return_value=[])

    def mock_is_allowed(url):
        return "private" not in url

    crawler.robots_parser.is_allowed = MagicMock(side_effect=mock_is_allowed)

    page_html = """
    <html><body>
        <h1>Home</h1>
        <p>""" + "words " * 40 + """</p>
        <a href="https://webandcrafts.com/private/secret">Private</a>
        <a href="https://webandcrafts.com/public">Public</a>
    </body></html>
    """

    async def mock_fetch(client, url, retries=2):
        return CrawledPage(url=url, canonical_url=url, html=page_html)

    crawler.fetch_page_httpx = AsyncMock(side_effect=mock_fetch)

    with patch("app.rag.crawler.crawler.SitemapParser.discover_urls", new=AsyncMock(return_value=[])):
        pages = await crawler.crawl(["https://webandcrafts.com/"])

    crawled_urls = [p.url for p in pages]
    assert "https://webandcrafts.com/private/secret" not in crawled_urls
    # But it was discovered via link extraction
    assert "https://webandcrafts.com/private/secret" in crawler.discovered_urls


@pytest.mark.anyio
async def test_crawl_service_metrics_and_error_handling():
    """Verify CrawlService stores accurate discovered_count from crawler.discovered_urls and handles errors."""
    mock_crawler = WebCrawler(allowed_domains=["webandcrafts.com"])
    mock_crawler.discovered_urls = {
        "https://webandcrafts.com/",
        "https://webandcrafts.com/a",
        "https://webandcrafts.com/b",
    }
    mock_crawler.visited_urls = {
        "https://webandcrafts.com/",
        "https://webandcrafts.com/a",
    }
    mock_crawler.failed_urls = {
        "https://webandcrafts.com/fail": "404 Not Found"
    }

    mock_page = CrawledPage(
        url="https://webandcrafts.com/a",
        canonical_url="https://webandcrafts.com/a",
        html="<html><body><h1>Test</h1><p>" + "meaningful content " * 30 + "</p></body></html>",
    )
    mock_crawler.crawl = AsyncMock(return_value=[mock_page])

    mock_crawl_repo = MagicMock()
    mock_crawl_repo.create = AsyncMock(return_value="run123")
    mock_crawl_repo.update = AsyncMock(return_value=True)
    mock_crawl_repo.mark_stale_runs = AsyncMock(return_value=0)

    mock_indexer = MagicMock()
    mock_indexer.index_document = AsyncMock(return_value=("indexed", "doc123", 3))

    service = CrawlService(
        crawler=mock_crawler,
        indexer=mock_indexer,
        crawl_repo=mock_crawl_repo,
    )

    run_id = await service.start_crawl(["https://webandcrafts.com/"])

    assert run_id == "run123"
    mock_crawl_repo.update.assert_awaited_once()
    call_args = mock_crawl_repo.update.call_args[0]
    update_dict = call_args[1]

    # Check metrics
    assert update_dict["status"] == "completed"
    assert update_dict["urls_discovered"] == 3
    assert update_dict["urls_crawled"] == 1
    assert update_dict["documents_changed"] == 1
    assert update_dict["chunks_created"] == 3
    # Crawler failure is recorded in errors
    assert any(e["url"] == "https://webandcrafts.com/fail" for e in update_dict["errors"])


@pytest.mark.anyio
async def test_crawl_service_crawl_level_exception():
    """Verify unexpected crawl-level exceptions mark CrawlRun as failed."""
    mock_crawler = MagicMock()
    mock_crawler.discovered_urls = {"https://webandcrafts.com/"}
    mock_crawler.crawl = AsyncMock(side_effect=RuntimeError("Fatal network crash"))

    mock_crawl_repo = MagicMock()
    mock_crawl_repo.create = AsyncMock(return_value="run_fatal")
    mock_crawl_repo.update = AsyncMock(return_value=True)
    mock_crawl_repo.mark_stale_runs = AsyncMock(return_value=0)

    service = CrawlService(
        crawler=mock_crawler,
        crawl_repo=mock_crawl_repo,
    )

    with pytest.raises(RuntimeError, match="Fatal network crash"):
        await service.start_crawl(["https://webandcrafts.com/"])

    # Verify run marked failed
    mock_crawl_repo.update.assert_awaited_once()
    call_args = mock_crawl_repo.update.call_args[0]
    update_dict = call_args[1]
    assert update_dict["status"] == "failed"
    assert any("Fatal network crash" in str(e) for e in update_dict["errors"])
