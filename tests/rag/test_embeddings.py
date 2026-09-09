import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from app.core.config import settings
from app.rag.embeddings.embedding_service import EmbeddingService
from app.rag.embeddings.providers import (
    BaseEmbeddingProvider,
    GeminiEmbeddingProvider,
    LocalEmbeddingProvider,
    EmbeddingProviderFactory,
)
from app.rag.exceptions import EmbeddingException, RAGConfigurationException
from app.ai.service import AIService
from app.ai.schemas import AIRequest, AIResponse
from app.rag.models import RAGResult
from app.services.rag_service import RAGService


# =========================================================================
# 1. PROVIDER CONSTRUCTION TESTS
# =========================================================================

def test_gemini_provider_construction():
    provider = GeminiEmbeddingProvider(
        model="gemini-embedding-001",
        dimensions=768,
        api_key="test-api-key",
    )
    assert provider.provider_name == "gemini"
    assert provider.model == "gemini-embedding-001"
    assert provider.dimensions == 768


def test_local_provider_construction():
    provider = LocalEmbeddingProvider(
        model="BAAI/bge-base-en-v1.5",
        dimensions=768,
    )
    assert provider.provider_name == "local"
    assert provider.model == "BAAI/bge-base-en-v1.5"
    assert provider.dimensions == 768


# =========================================================================
# 2. FACTORY SELECTION & CONFIGURATION TESTS
# =========================================================================

def test_provider_factory_selects_gemini():
    provider = EmbeddingProviderFactory.create("gemini", api_key="test-key")
    assert isinstance(provider, GeminiEmbeddingProvider)
    assert provider.provider_name == "gemini"


def test_provider_factory_selects_local():
    provider = EmbeddingProviderFactory.create("local")
    assert isinstance(provider, LocalEmbeddingProvider)
    assert provider.provider_name == "local"


def test_provider_factory_case_insensitive():
    provider = EmbeddingProviderFactory.create("LOCAL")
    assert isinstance(provider, LocalEmbeddingProvider)
    assert provider.provider_name == "local"


def test_provider_factory_invalid_provider_raises_error():
    with pytest.raises(RAGConfigurationException) as exc_info:
        EmbeddingProviderFactory.create("unsupported_provider")
    assert "Unsupported embedding provider 'unsupported_provider'" in str(exc_info.value)


# =========================================================================
# 3. LOCAL PROVIDER FUNCTIONALITY (MOCKED INFERENCE)
# =========================================================================

@pytest.mark.anyio
async def test_local_provider_single_embedding():
    provider = LocalEmbeddingProvider(dimensions=768)
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.05] * 768], dtype=np.float32)

    with patch.object(provider, "_get_model", return_value=mock_model):
        vec = await provider.get_embedding("WAC Enterprise Cloud Solutions")
        assert len(vec) == 768
        assert isinstance(vec[0], float)
        mock_model.encode.assert_called_once()
        args, kwargs = mock_model.encode.call_args
        assert args[0] == ["WAC Enterprise Cloud Solutions"]
        assert kwargs.get("normalize_embeddings") is True


@pytest.mark.anyio
async def test_local_provider_batch_embeddings():
    provider = LocalEmbeddingProvider(dimensions=768)
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768], dtype=np.float32)

    with patch.object(provider, "_get_model", return_value=mock_model):
        batch_vecs = await provider.get_batch_embeddings(["Chunk 1", "Chunk 2"])
        assert len(batch_vecs) == 2
        assert len(batch_vecs[0]) == 768
        assert len(batch_vecs[1]) == 768
        mock_model.encode.assert_called_once()


@pytest.mark.anyio
async def test_local_provider_dimension_mismatch_raises():
    provider = LocalEmbeddingProvider(dimensions=768)
    mock_model = MagicMock()
    # Return 512 dimensions instead of expected 768
    mock_model.encode.return_value = np.array([[0.1] * 512], dtype=np.float32)

    with patch.object(provider, "_get_model", return_value=mock_model):
        with pytest.raises(EmbeddingException) as exc_info:
            await provider.get_embedding("Dimension test")
        assert "dimension mismatch" in str(exc_info.value).lower()


# =========================================================================
# 4. GEMINI PROVIDER (MOCKED GOOGLE GENAI CLIENT)
# =========================================================================

@pytest.mark.anyio
async def test_gemini_provider_mocked_call():
    provider = GeminiEmbeddingProvider(
        model="gemini-embedding-001",
        dimensions=768,
        api_key="mock-gemini-key",
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_val = MagicMock()
    mock_val.values = [0.05] * 768
    mock_response.embeddings = [mock_val]
    mock_client.models.embed_content.return_value = mock_response

    with patch.object(GeminiEmbeddingProvider, "client", new=mock_client):
        vec = await provider.get_embedding("WAC Technology")
        assert len(vec) == 768
        mock_client.models.embed_content.assert_called_once()


@pytest.mark.anyio
async def test_gemini_provider_hard_quota_fails_immediately():
    """Verify daily/free-tier quota exhaustion fails immediately without retrying."""
    provider = GeminiEmbeddingProvider(
        model="gemini-embedding-001",
        dimensions=768,
        api_key="mock-key",
        default_retries=3,
    )
    mock_client = MagicMock()
    quota_exc = Exception("Quota exceeded for quota metric 'daily_requests' and limit per day exceeded")
    quota_exc.code = 429
    mock_client.models.embed_content.side_effect = quota_exc

    with patch.object(GeminiEmbeddingProvider, "client", new=mock_client), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        with pytest.raises(EmbeddingException) as exc_info:
            await provider.get_embedding("Query")

        assert "quota exhausted" in str(exc_info.value).lower()
        # Must fail on attempt 1 without sleeping or retrying 3 times
        assert mock_client.models.embed_content.call_count == 1
        assert mock_sleep.call_count == 0


@pytest.mark.anyio
async def test_gemini_provider_non_retryable_auth_fails_immediately():
    """Verify invalid API key (400/401) fails immediately without retrying."""
    provider = GeminiEmbeddingProvider(
        model="gemini-embedding-001",
        dimensions=768,
        api_key="invalid-key",
        default_retries=3,
    )
    mock_client = MagicMock()
    auth_exc = Exception("API_KEY_INVALID: API key not valid. Please pass a valid API key.")
    auth_exc.code = 400
    mock_client.models.embed_content.side_effect = auth_exc

    with patch.object(GeminiEmbeddingProvider, "client", new=mock_client), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        with pytest.raises(EmbeddingException) as exc_info:
            await provider.get_embedding("Query")

        assert "non-retryable" in str(exc_info.value).lower()
        assert mock_client.models.embed_content.call_count == 1
        assert mock_sleep.call_count == 0


# =========================================================================
# 5. REGRESSION: LOCAL PROVIDER MAKES ZERO GEMINI API CALLS
# =========================================================================

@pytest.mark.anyio
async def test_regression_local_provider_makes_zero_gemini_calls():
    """
    Verify that when RAG_EMBEDDING_PROVIDER is 'local',
    LocalEmbeddingProvider generates embeddings without touching Google GenAI.
    """
    with patch.object(settings, "RAG_EMBEDDING_PROVIDER", "local"):
        service = EmbeddingService()
        assert service.provider_name == "local"
        assert isinstance(service.provider_instance, LocalEmbeddingProvider)

        mock_local_model = MagicMock()
        mock_local_model.encode.return_value = np.array([[0.01] * 768], dtype=np.float32)

        with patch.object(service.provider_instance, "_get_model", return_value=mock_local_model), \
             patch("google.genai.Client") as mock_genai_client:

            vec = await service.get_embedding("What services does Web and Crafts provide?")
            assert len(vec) == 768
            mock_local_model.encode.assert_called_once()
            mock_genai_client.assert_not_called()


# =========================================================================
# 6. PROVIDER INDEPENDENCE MATRIX & INTEGRATION
# =========================================================================

@pytest.mark.parametrize("embedding_provider", ["local", "gemini"])
@pytest.mark.parametrize("generation_provider", ["gemini", "groq"])
def test_provider_configuration_independence_matrix(embedding_provider, generation_provider):
    """
    Ensure all 4 configuration combinations decouple embedding from generation:
    1. Local embedding + Gemini generation
    2. Local embedding + Groq generation
    3. Gemini embedding + Gemini generation
    4. Gemini embedding + Groq generation
    """
    with patch.object(settings, "RAG_EMBEDDING_PROVIDER", embedding_provider), \
         patch.object(settings, "DEFAULT_PROVIDER", generation_provider):

        # 1. Embedding service selects embedding provider correctly
        service = EmbeddingService(api_key="test-key" if embedding_provider == "gemini" else None)
        assert service.provider_name == embedding_provider

        # 2. AI generation provider remains configured independently
        ai_service = AIService()
        assert settings.DEFAULT_PROVIDER == generation_provider
        assert settings.RAG_EMBEDDING_PROVIDER == embedding_provider


@pytest.mark.anyio
@pytest.mark.parametrize("embedding_provider", ["local", "gemini"])
@pytest.mark.parametrize("generation_provider", ["gemini", "groq"])
async def test_aiservice_chat_integration_with_providers(embedding_provider, generation_provider):
    """
    Test AIService.chat end-to-end with all 4 provider combinations using mocked calls.
    """
    with patch.object(settings, "RAG_EMBEDDING_PROVIDER", embedding_provider), \
         patch.object(settings, "DEFAULT_PROVIDER", generation_provider):

        ai_service = AIService()

        mock_rag_result = RAGResult(
            is_relevant=True,
            has_context=True,
            context="Web and Crafts provides digital engineering and custom software.",
            sources=[{"title": "WAC Services", "url": "https://webandcrafts.com/services"}],
            confidence_score=0.95,
        )

        mock_gen_provider = AsyncMock()
        mock_gen_provider.generate = AsyncMock(
            return_value=AIResponse(
                content="Web and Crafts provides digital engineering.",
                tokens_used=42,
            )
        )

        with patch.object(RAGService, "get_grounded_context", new=AsyncMock(return_value=mock_rag_result)), \
             patch("app.ai.factory.ProviderFactory.get_provider", return_value=mock_gen_provider):

            response, rag_res = await ai_service.chat(
                message="Tell me about WAC engineering services.",
                provider=generation_provider,
            )

            assert response.content == "Web and Crafts provides digital engineering."
            assert rag_res.has_context is True
            assert len(rag_res.sources) == 1
            mock_gen_provider.generate.assert_called_once()
