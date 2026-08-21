import pytest
from app.rag.crawler.sitemap import SitemapParser


def test_sitemap_xml_parsing():
    parser = SitemapParser()
    sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
   <url>
      <loc>https://webandcrafts.com/services</loc>
      <lastmod>2026-08-01T10:00:00Z</lastmod>
   </url>
   <url>
      <loc>https://webandcrafts.com/about</loc>
      <lastmod>2026-07-15T08:00:00Z</lastmod>
   </url>
</urlset>"""

    parsed_urls = parser.parse_xml_content(sitemap_xml)
    assert len(parsed_urls) == 2
    assert "https://webandcrafts.com/services" in parsed_urls
    assert "https://webandcrafts.com/about" in parsed_urls


def test_sitemap_index_xml_parsing():
    parser = SitemapParser()
    sitemap_index_xml = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
   <sitemap>
      <loc>https://webandcrafts.com/sitemap-pages.xml</loc>
   </sitemap>
   <sitemap>
      <loc>https://webandcrafts.com/sitemap-posts.xml</loc>
   </sitemap>
</sitemapindex>"""

    child_sitemaps = parser.parse_xml_content(sitemap_index_xml)
    assert len(child_sitemaps) == 2
    assert "https://webandcrafts.com/sitemap-pages.xml" in child_sitemaps
    assert "https://webandcrafts.com/sitemap-posts.xml" in child_sitemaps
