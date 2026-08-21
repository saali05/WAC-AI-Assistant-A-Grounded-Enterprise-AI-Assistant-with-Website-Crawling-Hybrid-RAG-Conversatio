import ipaddress
import re
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode, urljoin
from app.core.config import settings
from app.rag.exceptions import SSRFProtectionException


# Common tracking parameters to strip during URL normalization
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "ref", "source", "_ga", "_gl",
    "mc_cid", "mc_eid", "yclid", "_hsenc", "_hsmi"
}

# Private / reserved IP networks for SSRF protection
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


class URLNormalizer:
    """URL normalization, canonicalization, and security validation (SSRF & domain allowlist)."""

    @staticmethod
    def is_ssrf_risk(hostname: str) -> bool:
        """Check if hostname resolves to local/private IP address or metadata endpoint."""
        if not hostname:
            return True

        host_lower = hostname.lower()

        # Reject localhost & known metadata hosts
        if host_lower in ("localhost", "localhost.localdomain", "metadata.google.internal", "169.254.169.254"):
            return True

        # Check IP address strings directly
        try:
            ip = ipaddress.ip_address(host_lower)
            for net in BLOCKED_IP_NETWORKS:
                if ip in net:
                    return True
        except ValueError:
            pass  # Hostname is a domain name, not a literal IP address

        return False

    @classmethod
    def validate_domain(cls, url: str, allowed_domains: list[str] | None = None) -> bool:
        """Verify URL uses allowed domain and protocol."""
        if not url:
            return False

        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https"):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        if cls.is_ssrf_risk(hostname):
            return False

        domains = allowed_domains or settings.allowed_domains_list
        host_lower = hostname.lower()

        for allowed in domains:
            allowed_lower = allowed.lower()
            if host_lower == allowed_lower or host_lower.endswith("." + allowed_lower):
                return True

        return False

    @classmethod
    def normalize(cls, url: str, base_url: str | None = None) -> str:
        """
        Normalize and canonicalize URL:
        - Resolve relative against base_url if supplied
        - Lowercase scheme & hostname
        - Strip tracking parameters & fragment
        - Normalize default ports
        - Normalize trailing slashes
        - Validate allowed domain & SSRF safety
        """
        if not url or not isinstance(url, str):
            raise SSRFProtectionException("Empty or invalid URL provided.")

        url_str = url.strip()

        # Check for disallowed URI schemes upfront
        if url_str.startswith(("javascript:", "file:", "data:", "mailto:", "tel:")):
            raise SSRFProtectionException(f"Unsupported URI scheme in URL: {url_str}")

        # Resolve relative URLs
        if base_url:
            url_str = urljoin(base_url, url_str)

        parsed = urlparse(url_str)

        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            raise SSRFProtectionException(f"Invalid URL scheme '{scheme}'. Only HTTP and HTTPS are allowed.")

        hostname = parsed.hostname
        if not hostname or cls.is_ssrf_risk(hostname):
            raise SSRFProtectionException(f"SSRF violation or invalid hostname: '{hostname}'")

        if not cls.validate_domain(url_str):
            raise SSRFProtectionException(f"Domain '{hostname}' is not in RAG_ALLOWED_DOMAINS allowlist.")

        # Normalize port
        port = parsed.port
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            netloc = hostname.lower()
        elif port:
            netloc = f"{hostname.lower()}:{port}"
        else:
            netloc = hostname.lower()

        # Normalize path
        path = parsed.path
        if not path:
            path = "/"
        elif path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        # Strip tracking query params
        query_dict = parse_qs(parsed.query, keep_blank_values=False)
        filtered_query = {
            k: v for k, v in query_dict.items()
            if k.lower() not in TRACKING_PARAMS
        }

        # Rebuild query string with sorted keys for consistency
        sorted_params = []
        for k in sorted(filtered_query.keys()):
            for v in sorted(filtered_query[k]):
                sorted_params.append((k, v))

        query_str = urlencode(sorted_params)

        normalized = urlunparse((
            scheme,
            netloc,
            path,
            parsed.params,
            query_str,
            ""  # Always strip fragment
        ))

        return normalized
