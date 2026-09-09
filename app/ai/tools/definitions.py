from google.genai import types

SEARCH_WAC_KNOWLEDGE = types.FunctionDeclaration(
    name="search_wac_knowledge",
    description=(
        "Search the official Web and Craft knowledge base for factual "
        "information about Web and Craft services, technologies, "
        "solutions, company information, careers, case studies, "
        "projects, industries, and other WAC-specific information. "
        "Use this function whenever the user's question requires "
        "specific or factual information about Web and Craft."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A precise search query describing the Web and Craft "
                    "information needed to answer the user's question."
                ),
            }
        },
        "required": ["query"],
    },
)


WAC_KNOWLEDGE_TOOL = types.Tool(
    function_declarations=[
        SEARCH_WAC_KNOWLEDGE,
    ]
)


WAC_TOOLS = [
    WAC_KNOWLEDGE_TOOL,
]