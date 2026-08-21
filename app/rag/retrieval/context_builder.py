from app.rag.models import RetrievedChunk, SourceCitation


class ContextBuilder:
    """Constructs dynamic WAC knowledge context blocks and source citations for LLM prompt."""

    @staticmethod
    def build_context_and_sources(chunks: list[RetrievedChunk]) -> tuple[str, list[SourceCitation]]:
        """Format top-K reranked chunks into clean context block and citation objects."""
        if not chunks:
            return "", []

        context_blocks: list[str] = ["WAC KNOWLEDGE CONTEXT\n"]
        sources: list[SourceCitation] = []
        seen_sources: set[str] = set()

        for idx, chunk in enumerate(chunks, start=1):
            heading = " > ".join(chunk.heading_path) if chunk.heading_path else ""

            block = (
                f"Source {idx}:\n"
                f"Title: {chunk.title}\n"
                f"URL: {chunk.url}\n"
            )
            if heading:
                block += f"Section: {heading}\n"
            block += f"\nContent:\n{chunk.content}\n"

            context_blocks.append(block)

            # Deduplicate sources by URL + heading
            source_key = f"{chunk.url}#{heading}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append(
                    SourceCitation(
                        title=chunk.title,
                        url=chunk.url,
                        heading=heading or None,
                        score=chunk.score,
                        canonical_url=chunk.canonical_url
                    )
                )

        full_context = "\n----------------------------------------\n".join(context_blocks)
        return full_context, sources
