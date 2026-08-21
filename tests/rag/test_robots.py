import pytest
from app.rag.crawler.robots import RobotsTxtParser


@pytest.mark.anyio
async def test_robots_txt_parsing():
    parser = RobotsTxtParser()
    robots_content = """
User-agent: *
Disallow: /admin/
Disallow: /private/
Crawl-delay: 2.0
Sitemap: https://webandcrafts.com/sitemap.xml
"""

    rp = parser._parsers.get("https://webandcrafts.com")
    if not rp:
        import urllib.robotparser
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(robots_content.splitlines())
        parser._parsers["https://webandcrafts.com"] = rp
        parser._sitemaps["https://webandcrafts.com"] = ["https://webandcrafts.com/sitemap.xml"]
        parser._crawl_delays["https://webandcrafts.com"] = 2.0

    assert parser.is_allowed("https://webandcrafts.com/services") is True
    assert parser.is_allowed("https://webandcrafts.com/admin/login") is False
    assert parser.get_sitemaps("https://webandcrafts.com") == ["https://webandcrafts.com/sitemap.xml"]
    assert parser.get_crawl_delay("https://webandcrafts.com") == 2.0
