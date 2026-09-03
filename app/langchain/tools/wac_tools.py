from langchain_core.tools import tool

from app.langchain.retrievers.wac_retriever import WACRetriever
from app.core.logging import logger


retriever = WACRetriever()


@tool
async def search_wac_knowledge(query: str) -> str:
    """
    Search the WAC knowledge base for information about
    Web and Craft, including services, technologies,
    projects, careers, clients, and company information.

    Use this tool whenever the user asks for information
    that requires factual knowledge about WAC.
    """

    if not query or not query.strip():
        return "No search query was provided."

    logger.info(
        "LangChain WAC tool called | query=%s",
        query,
    )

    documents = await retriever.ainvoke(query)

    if not documents:
        return (
            "No relevant information was found "
            "in the WAC knowledge base."
        )

    results = []

    for index, document in enumerate(documents, start=1):

        title = document.metadata.get(
            "title",
            "",
        )

        url = document.metadata.get(
            "url",
            "",
        )

        score = document.metadata.get(
            "score",
            "",
        )

        results.append(
            f"""
Source {index}
Title: {title}
URL: {url}
Score: {score}

Content:
{document.page_content}
"""
        )

    return "\n".join(results)