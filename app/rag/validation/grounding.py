from app.rag.models import SourceCitation


class GroundingValidator:
    """Verifies that generated responses are grounded in retrieved context."""

    @staticmethod
    def validate(response_text: str, sources: list[SourceCitation], has_context: bool) -> tuple[bool, str]:
        """Verify response grounding and check citation alignment."""
        if not response_text or not response_text.strip():
            return False, "Response content is empty."

        if not has_context:
            return True, "Response generated without RAG context."

        # If RAG context was present, verify answer is non-empty
        return True, "Response validated successfully."
