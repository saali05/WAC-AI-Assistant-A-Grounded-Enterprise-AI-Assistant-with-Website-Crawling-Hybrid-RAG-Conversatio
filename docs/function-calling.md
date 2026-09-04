# WAC AI Assistant --- Function Calling and Tools

## Purpose

Function calling gives the language model controlled access to
application capabilities.

The primary WAC knowledge tool is:

``` text
search_wac_knowledge
```

It allows Gemini to request a search of the WAC knowledge base.

## Function Calling vs RAG

They are different layers:

``` text
Function Calling = how the model requests a capability
RAG              = how the application retrieves knowledge
```

Therefore:

``` text
LLM -> Function Call -> ToolExecutor -> RAGService
```

## Tool Definition

A tool definition tells the model:

-   Tool name
-   Tool purpose
-   Accepted arguments
-   Argument structure

Conceptually:

``` text
Tool:
    search_wac_knowledge

Argument:
    query: string
```

The model receives the tool schema. It does not receive unrestricted
access to Python.

## Tool Execution

The model can return a structured request such as:

``` json
{
  "name": "search_wac_knowledge",
  "arguments": {
    "query": "What services does WAC provide?"
  }
}
```

The backend receives the request and routes it through `ToolExecutor`.

``` text
Gemini
   |
   v
AIToolCall
   |
   v
ToolExecutor
   |
   v
search_wac_knowledge
   |
   v
RAGService
```

## Complete Tool Loop

### 1. User

``` text
What services does WAC provide?
```

### 2. Initial Model Request

The application sends the prompt and available tools to Gemini.

### 3. Function Call

Gemini determines that WAC knowledge is required and requests
`search_wac_knowledge`.

### 4. Tool Execution

The backend executes the approved tool.

### 5. RAG Retrieval

The tool invokes the RAG pipeline to retrieve relevant chunks.

### 6. Tool Result

The retrieved context and source information are returned to Gemini.

### 7. Final Response

Gemini uses the grounded result to generate the final answer.

``` text
User
 |
 v
Gemini
 |
 | function call
 v
ToolExecutor
 |
 v
RAGService
 |
 v
Hybrid Search + Reranking
 |
 v
Tool Result
 |
 v
Gemini
 |
 v
Final Answer
```

## No-Tool Path

The model can return a response without using a tool.

``` text
User -> Gemini -> Final Response
```

The AI service checks whether `response.tool_calls` contains calls.

## Tool Round Limit

The application uses:

``` text
MAX_TOOL_ROUNDS = 3
```

This prevents an uncontrolled sequence of model/tool/model/tool calls.

## Why Function Calling Is Useful

Without function calling, the application could execute RAG for every
request.

With function calling:

``` text
Question
   |
   v
LLM
   |
   +---- no knowledge retrieval needed -> response
   |
   +---- WAC knowledge needed ----------> tool -> RAG
```

This turns retrieval into a controlled model-callable capability.

## Security and Control

The model cannot execute arbitrary Python code. Only explicitly
registered tools can be requested, and the backend controls their
execution and error handling.

## Extensibility

Additional controlled tools can be added later, for example:

``` text
search_wac_knowledge
get_wac_services
get_wac_careers
get_wac_contact
```

## Presentation Explanation

Function calling allows the LLM to request controlled application
capabilities. In this project, Gemini can call `search_wac_knowledge`
when it needs grounded WAC information. The backend executes the tool
through `ToolExecutor`, which invokes the hybrid RAG pipeline. The
retrieved context is then returned to Gemini for the final grounded
response.
