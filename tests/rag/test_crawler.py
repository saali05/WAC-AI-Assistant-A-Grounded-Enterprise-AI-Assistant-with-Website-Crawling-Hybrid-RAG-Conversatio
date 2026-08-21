import pytest
from app.rag.crawler.crawler import WebCrawler, CrawledPage


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
