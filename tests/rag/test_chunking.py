import pytest
from app.rag.chunking.semantic_chunker import SemanticChunker
from app.rag.extraction.html_extractor import ExtractedHTML, HeadingSection


def test_semantic_chunker_metadata_preservation():
    chunker = SemanticChunker(target_chunk_size=100)

    extracted = ExtractedHTML(
        title="WAC AI Solutions",
        description="AI Services",
        canonical_url="https://webandcrafts.com/ai",
        main_text="Content",
        sections=[
            HeadingSection(
                heading_level=1,
                heading_title="AI Solutions",
                heading_path=["AI Solutions"],
                paragraphs=["Web and Crafts designs enterprise generative AI systems.", "We specialize in LLM workflows."]
            ),
            HeadingSection(
                heading_level=2,
                heading_title="Vector Search",
                heading_path=["AI Solutions", "Vector Search"],
                paragraphs=["Our vector search systems leverage MongoDB Atlas vector search."]
            )
        ]
    )

    chunks = chunker.chunk_document(
        document_id="doc1",
        extracted=extracted,
        url="https://webandcrafts.com/ai",
        canonical_url="https://webandcrafts.com/ai"
    )

    assert len(chunks) >= 2
    assert chunks[0].heading_path == ["AI Solutions"]
    assert chunks[1].heading_path == ["AI Solutions", "Vector Search"]
    assert chunks[0].canonical_url == "https://webandcrafts.com/ai"
