from typing import Any


WAC_LIVE_TOOLS: list[dict[str, Any]] = [
    {
        "function_declarations": [
            {
                "name": "search_wac_knowledge",
                "description": (
                    "Search the official WAC knowledge base for factual "
                    "information about Web and Craft, including services, "
                    "technologies, solutions, projects, industries, clients, "
                    "case studies, careers, leadership, offices and company information. "
                    "Use this tool whenever the user asks for factual WAC-related "
                    "information. Do not use general world knowledge to answer WAC questions."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": (
                                "A precise search query describing the WAC "
                                "information the user wants."
                            ),
                        }
                    },
                    "required": ["query"],
                },
            }
        ]
    }
]