import pytest
from app.rag.crawler.url_normalizer import URLNormalizer
from app.rag.exceptions import SSRFProtectionException


def test_url_normalization_basic():
    url = "HTTPS://WWW.WebAndCrafts.COM/services/?utm_source=google&fbclid=123#section1"
    normalized = URLNormalizer.normalize(url)
    assert normalized == "https://www.webandcrafts.com/services"


def test_url_normalization_trailing_slash():
    url = "https://webandcrafts.com/about/"
    normalized = URLNormalizer.normalize(url)
    assert normalized == "https://webandcrafts.com/about"


def test_relative_url_resolution():
    base = "https://webandcrafts.com/services"
    rel = "/careers?gclid=xyz"
    normalized = URLNormalizer.normalize(rel, base_url=base)
    assert normalized == "https://webandcrafts.com/careers"


def test_domain_allowlist_validation():
    allowed = URLNormalizer.validate_domain("https://webandcrafts.com/contact")
    assert allowed is True

    subdomain_allowed = URLNormalizer.validate_domain("https://www.webandcrafts.com/contact")
    assert subdomain_allowed is True

    external_domain = URLNormalizer.validate_domain("https://google.com")
    assert external_domain is False


def test_ssrf_protection_rejection():
    # Localhost rejection
    with pytest.raises(SSRFProtectionException):
        URLNormalizer.normalize("http://localhost:8000/admin")

    # 127.0.0.1 IP rejection
    with pytest.raises(SSRFProtectionException):
        URLNormalizer.normalize("http://127.0.0.1/internal")

    # Private network 192.168.x rejection
    with pytest.raises(SSRFProtectionException):
        URLNormalizer.normalize("http://192.168.1.1/secret")

    # Cloud metadata endpoint rejection
    with pytest.raises(SSRFProtectionException):
        URLNormalizer.normalize("http://169.254.169.254/latest/meta-data")

    # Unsupported scheme rejection
    with pytest.raises(SSRFProtectionException):
        URLNormalizer.normalize("file:///etc/passwd")

    with pytest.raises(SSRFProtectionException):
        URLNormalizer.normalize("javascript:alert(1)")
