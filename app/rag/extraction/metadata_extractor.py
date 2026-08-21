from urllib.parse import urlparse
from app.rag.extraction.content_cleaner import ContentCleaner
from app.rag.extraction.html_extractor import HTMLExtractor, ExtractedHTML
from app.rag.models import RAGDocumentModel


class MetadataExtractor:
    """Metadata extraction and document quality validation."""

    @classmethod
    def extract_document_model(
        cls,
        raw_html: str,
        url: str,
        http_status: int = 200
    ) -> tuple[RAGDocumentModel, ExtractedHTML]:
        """Extract metadata and build RAGDocumentModel instance."""
        extracted = HTMLExtractor.extract(raw_html, url)
        cleaned_text = extracted.main_text
        content_hash = ContentCleaner.calculate_content_hash(cleaned_text)

        parsed_url = urlparse(url)
        domain = parsed_url.hostname.lower() if parsed_url.hostname else ""

        words = [w for w in cleaned_text.split() if w.strip()]
        word_count = len(words)

        doc_model = RAGDocumentModel(
            url=url,
            canonical_url=extracted.canonical_url or url,
            title=extracted.title or domain,
            description=extracted.description,
            content_hash=content_hash,
            domain=domain,
            word_count=word_count,
            http_status=http_status,
            status="active" if cls.validate_quality(cleaned_text, http_status) else "inactive"
        )

        return doc_model, extracted

    @staticmethod
    def validate_quality(text: str, http_status: int = 200, min_word_count: int = 30) -> bool:
        """Validate whether page content is sufficient and meaningful for indexing."""
        if http_status != 200:
            return False

        if not text or not text.strip():
            return False

        words = [w for w in text.split() if w.strip()]
        if len(words) < min_word_count:
            return False

        # Reject login or error pages based on common indicators
        lower_text = text.lower()
        if "404 not found" in lower_text or "access denied" in lower_text or "page not found" in lower_text:
            return False

        return True
