import pytest
from app.rag.extraction.content_cleaner import ContentCleaner
from app.rag.extraction.html_extractor import HTMLExtractor
from app.rag.extraction.metadata_extractor import MetadataExtractor


SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>WAC AI &amp; Intelligence Services</title>
    <meta name="description" content="Leading AI and custom software solutions by Web and Crafts.">
    <link rel="canonical" href="https://webandcrafts.com/services/ai">
</head>
<body>
    <nav class="navbar">
        <a href="/">Home</a>
        <a href="/services">Services</a>
    </nav>

    <main>
        <h1>AI &amp; Intelligence Solutions</h1>
        <p>Web and Crafts delivers state-of-the-art enterprise artificial intelligence applications.</p>

        <h2>Generative AI</h2>
        <p>We build specialized RAG systems, LLM agents, and custom neural search engines.</p>

        <h3>Key Capabilities</h3>
        <ul>
            <li>Vector Search Optimization</li>
            <li>Custom Fine-Tuning</li>
        </ul>
    </main>

    <footer class="footer-container">
        <p>&copy; 2026 Web and Crafts. All rights reserved.</p>
    </footer>
</body>
</html>
"""


def test_content_cleaner_boilerplate_removal():
    clean_html = ContentCleaner.clean(SAMPLE_HTML)
    assert "<nav" not in clean_html
    assert "<footer" not in clean_html

    plain_text = ContentCleaner.extract_plain_text(clean_html)
    assert "AI & Intelligence Solutions" in plain_text
    assert "Web and Crafts delivers state-of-the-art enterprise" in plain_text
    assert "&copy;" not in plain_text


def test_sha256_hash_calculation():
    text1 = "Web and Crafts delivers AI services."
    text2 = "Web and Crafts delivers AI services."
    text3 = "Different text content."

    hash1 = ContentCleaner.calculate_content_hash(text1)
    hash2 = ContentCleaner.calculate_content_hash(text2)
    hash3 = ContentCleaner.calculate_content_hash(text3)

    assert len(hash1) == 64
    assert hash1 == hash2
    assert hash1 != hash3


def test_html_extractor_metadata_and_hierarchy():
    extracted = HTMLExtractor.extract(SAMPLE_HTML, url="https://webandcrafts.com/services/ai")

    assert extracted.title == "WAC AI & Intelligence Services"
    assert extracted.description == "Leading AI and custom software solutions by Web and Crafts."
    assert extracted.canonical_url == "https://webandcrafts.com/services/ai"
    assert len(extracted.sections) >= 2

    # Verify heading hierarchy paths
    h2_section = [s for s in extracted.sections if s.heading_title == "Generative AI"][0]
    assert h2_section.heading_level == 2
    assert "AI & Intelligence Solutions" in h2_section.heading_path


def test_metadata_extractor_quality_validation():
    doc_model, extracted = MetadataExtractor.extract_document_model(SAMPLE_HTML, url="https://webandcrafts.com/services/ai")

    assert doc_model.canonical_url == "https://webandcrafts.com/services/ai"
    assert doc_model.status == "active"
    assert doc_model.word_count > 30

    # Short/empty content quality rejection test
    is_valid = MetadataExtractor.validate_quality("Short text", min_word_count=30)
    assert is_valid is False
