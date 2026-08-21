import re
from typing import Optional
from app.core.config import settings
from app.rag.extraction.html_extractor import ExtractedHTML, HeadingSection
from app.rag.extraction.content_cleaner import ContentCleaner
from app.rag.models import RAGChunkModel


class SemanticChunker:
    """Heading-aware semantic chunker for RAG document splitting."""

    def __init__(
        self,
        target_chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> None:
        # Approximate tokens as ~4 chars per token
        self.chunk_tokens = target_chunk_size or settings.RAG_CHUNK_SIZE
        self.overlap_tokens = chunk_overlap or settings.RAG_CHUNK_OVERLAP
        self.max_chars = self.chunk_tokens * 4
        self.overlap_chars = self.overlap_tokens * 4

    def chunk_document(
        self,
        document_id: str,
        extracted: ExtractedHTML,
        url: str,
        canonical_url: str,
        version: int = 1
    ) -> list[RAGChunkModel]:
        """Split ExtractedHTML into contextual, heading-aware RAGChunkModel instances."""
        chunks: list[RAGChunkModel] = []
        chunk_index = 0

        sections = extracted.sections
        if not sections:
            # Fallback if no sections extracted
            sections = [HeadingSection(heading_level=0, heading_title=extracted.title or "Page Content", heading_path=[], paragraphs=[extracted.main_text])]

        for section in sections:
            heading_title = section.heading_title or extracted.title
            heading_path = section.heading_path or ([heading_title] if heading_title else [])

            paragraphs = section.paragraphs
            if not paragraphs:
                continue

            current_chunk_text = ""
            current_paragraphs: list[str] = []

            for p in paragraphs:
                clean_p = p.strip()
                if not clean_p:
                    continue

                if len(current_chunk_text) + len(clean_p) <= self.max_chars:
                    current_paragraphs.append(clean_p)
                    current_chunk_text = "\n\n".join(current_paragraphs)
                else:
                    if current_chunk_text:
                        content_hash = ContentCleaner.calculate_content_hash(current_chunk_text)
                        chunks.append(RAGChunkModel(
                            document_id=document_id,
                            chunk_index=chunk_index,
                            content=current_chunk_text,
                            title=extracted.title or heading_title,
                            heading_path=heading_path,
                            url=url,
                            canonical_url=canonical_url,
                            embedding_model=settings.RAG_EMBEDDING_MODEL,
                            embedding_dimensions=settings.RAG_EMBEDDING_DIMENSIONS,
                            content_hash=content_hash,
                            version=version,
                            status="active"
                        ))
                        chunk_index += 1

                    # Overlap handling
                    overlap_text = current_chunk_text[-self.overlap_chars:] if len(current_chunk_text) > self.overlap_chars else ""
                    if overlap_text and len(clean_p) <= self.max_chars:
                        current_paragraphs = [overlap_text, clean_p]
                        current_chunk_text = "\n\n".join(current_paragraphs)
                    else:
                        current_paragraphs = [clean_p]
                        current_chunk_text = clean_p

            if current_chunk_text:
                content_hash = ContentCleaner.calculate_content_hash(current_chunk_text)
                chunks.append(RAGChunkModel(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=current_chunk_text,
                    title=extracted.title or heading_title,
                    heading_path=heading_path,
                    url=url,
                    canonical_url=canonical_url,
                    embedding_model=settings.RAG_EMBEDDING_MODEL,
                    embedding_dimensions=settings.RAG_EMBEDDING_DIMENSIONS,
                    content_hash=content_hash,
                    version=version,
                    status="active"
                ))
                chunk_index += 1

        return chunks
