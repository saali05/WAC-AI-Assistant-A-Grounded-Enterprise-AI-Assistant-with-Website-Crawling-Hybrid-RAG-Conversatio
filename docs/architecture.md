# WAC AI Assistant — Comprehensive System Architecture

## 1. High-Level Architecture Overview

The WAC AI Assistant is structured into modular layers spanning frontend presentation, API routing, intelligence services, retrieval pipelines, and persistence.

```mermaid
flowchart TD
    User([User Browser])
    
    subgraph Frontend["React 19 + TypeScript Frontend"]
        ChatUI[Chat Window & Model Selector]
        VoiceUI[Gemini Live Voice Interface]
        AnalyticsUI[Session Analytics Drawer]
    end
    
    subgraph API["FastAPI Application Gateway"]
        ChatRouter["/chat"]
        VoiceRouter["/voice (token, tool, message)"]
        AnalyticsRouter["/conversations/{id}/analytics"]
        CrawlRouter["/rag (crawl, reindex)"]
    end
    
    subgraph CoreServices["Core Services Layer"]
        ChatService[ChatService]
        AIService[AIService]
        UsageService[UsageService]
        RAGService[RAGService]
        WACRetriever[WACRetriever / LangChain]
    end
    
    subgraph AIProviders["AI Provider Layer"]
        GeminiGen[Gemini 2.5 Flash]
        GroqGen[Groq Llama 3.3 70B]
        GeminiLive[Gemini Live WebSocket]
    end
    
    subgraph EmbeddingLayer["Embedding Abstraction"]
        GeminiEmbed[GeminiEmbeddingProvider]
        LocalEmbed[LocalEmbeddingProvider / BGE]
    end
    
    subgraph Storage["MongoDB Persistence"]
        DB_Conv[(conversations / messages)]
        DB_RAG[(rag_documents / rag_chunks)]
        DB_Usage[(chat_usage / crawl_runs)]
    end

    User <--> Frontend
    Frontend <--> API
    API --> CoreServices
    CoreServices --> AIProviders
    CoreServices --> EmbeddingLayer
    CoreServices --> Storage
```

---

## 2. End-to-End Text Chat Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Frontend as React Client
    participant API as FastAPI /chat
    participant ChatService as ChatService
    participant Gate as WACRelevanceGate
    participant RAG as RAGService
    participant AI as AIService
    participant Provider as Gemini / Groq Provider
    participant Usage as UsageService
    participant DB as MongoDB

    User->>Frontend: Submit Message ("What services does WAC provide?")
    Frontend->>API: POST /chat {message, model: "gemini"}
    API->>ChatService: send_message(provider, message, conv_id)
    ChatService->>Gate: evaluate(message)
    
    alt Non-WAC Domain Query
        Gate-->>ChatService: is_wac=False, refusal_text
        ChatService-->>Frontend: Standard Refusal (0 Tokens Billed)
    else Valid WAC Query
        Gate-->>ChatService: is_wac=True
        ChatService->>RAG: get_grounded_context(message)
        RAG-->>ChatService: RAGResult(context, sources, score)
        ChatService->>AI: chat(message, context, provider)
        AI->>Provider: generate(prompt_with_grounding)
        Provider-->>AI: AIResponse(content, tokens_used)
        AI->>Usage: record_usage(conversation_id, AIUsage)
        Usage->>DB: Insert chat_usage record
        ChatService->>DB: Save user & assistant messages
        ChatService-->>Frontend: {response, sources, conversation_id}
        Frontend-->>User: Render Glassmorphic Bubble + Source Cards
    end
```

---

## 3. Detailed RAG Retrieval Flow

```mermaid
flowchart TD
    Query([User Query]) --> Gate{WAC Relevance Gate}
    
    Gate -- Non-WAC --> Refusal[Immediate Grounded Refusal]
    Gate -- WAC Query --> Rewriter[Query Rewriter & Intent Classifier]
    
    Rewriter --> RewriteDone[Expanded Query + Detected Intent]
    
    RewriteDone --> VectorSearch[Atlas Vector Search 768d]
    RewriteDone --> KeywordSearch[MongoDB Full-Text Search]
    
    VectorSearch --> RRF[Reciprocal Rank Fusion k=60]
    KeywordSearch --> RRF
    
    RRF --> Reranker[Fusion Reranker & URL Diversity Filter]
    
    Reranker --> SufficiencyGate{Evidence Sufficiency Gate}
    
    SufficiencyGate -- Generic Blog Only / No Company Attribution --> NoProofRefusal[Controlled Refusal: Insufficient Proof]
    SufficiencyGate -- Sufficient Company Grounding --> ContextBuilder[Context Builder & Source Formatter]
    
    ContextBuilder --> LLM[Grounded LLM Generation]
    LLM --> FinalAnswer([Final Answer + Verified Citations])
```

---

## 4. Gemini Live Multimodal Voice Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Frontend as GeminiLiveService (Web Audio)
    participant Backend as FastAPI Backend
    participant GeminiLive as Gemini Live WebSocket
    participant Retriever as WACRetriever / RAG
    participant Usage as UsageService

    User->>Frontend: Click Voice Button
    Frontend->>Backend: GET /voice/token
    Backend-->>Frontend: Ephemeral Token + WAC_LIVE_TOOLS schema
    Frontend->>GeminiLive: Connect WebSocket (Audio/PCM 16kHz)
    
    User->>Frontend: Speaks Query into Microphone
    Frontend->>GeminiLive: Stream PCM Chunks
    
    alt Knowledge Required
        GeminiLive->>Frontend: toolCall: search_wac_knowledge(query)
        Frontend->>Backend: POST /voice/tool {name, arguments}
        Backend->>Retriever: ainvoke(query)
        Retriever-->>Backend: Grounded Documents + Citations
        Backend-->>Frontend: Tool Result Payload
        Frontend->>GeminiLive: sendToolResponse(result)
    end
    
    GeminiLive->>Frontend: Model Turn Parts (Spoken Audio + Transcripts)
    Frontend->>User: Play 24kHz Web Audio Buffer
    
    GeminiLive->>Frontend: turnComplete
    Frontend->>Backend: POST /voice/message {user_message, assistant_message, audio_durations, tokens, session_id}
    Backend->>Usage: record_usage(voice_usage)
    Backend-->>Frontend: 200 OK (Persisted)
```

---

## 5. Analytics & Cost Accounting Flow

```mermaid
flowchart LR
    subgraph ExecutionTriggers["Execution Events"]
        ChatTurn[Text Chat Turn]
        VoiceTurn[Voice Live Turn]
        LangChainTurn[LangChain Pipeline Turn]
    end

    subgraph TelemetryExtraction["Telemetry Engine"]
        TokenCounter[Token Counter / Usage Metadata]
        AudioTracker[Web Audio Duration Tracker]
        PricingModel[Pricing Calculator MODEL_PRICING]
    end

    subgraph Database["MongoDB Storage"]
        ChatUsageCollection[("chat_usage Collection")]
    end

    subgraph AnalyticsEndpoint["Analytics Aggregation"]
        UsageServiceAPI["UsageService.get_session_analytics()"]
        FrontendSidebar["Session Analytics Drawer UI"]
    end

    ExecutionTriggers --> TelemetryExtraction
    TelemetryExtraction --> PricingModel
    PricingModel --> ChatUsageCollection
    ChatUsageCollection --> UsageServiceAPI
    UsageServiceAPI --> FrontendSidebar
```

---

## 6. Provider Decoupling Matrix

The architecture strictly isolates **Text Generation** from **Embedding Generation**:

```mermaid
flowchart TD
    subgraph GenerationLayer["Generation Provider (Configurable via Frontend & Env)"]
        GeminiGen["Gemini 2.5 Flash (Google GenAI)"]
        GroqGen["Llama 3.3 70B (Groq Cloud)"]
    end

    subgraph EmbeddingLayer["Embedding Provider (Configurable via RAG_EMBEDDING_PROVIDER)"]
        GeminiEmb["gemini-embedding-001 (Google API)"]
        LocalBGE["BAAI/bge-base-en-v1.5 (Local CPU/CUDA)"]
    end

    GeminiGen <.-> GeminiEmb
    GeminiGen <.-> LocalBGE
    GroqGen <.-> GeminiEmb
    GroqGen <.-> LocalBGE
```
