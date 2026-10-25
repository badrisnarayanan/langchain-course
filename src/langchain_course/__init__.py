from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient
from typing import List
from pydantic import BaseModel, Field

load_dotenv()  # Replace with your actual Tavily API key

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """Searches the web for a query and returns the results.

    Args:
        query (str): The query to search for.. no site terms.

    Returns:
        str: search results from web
    """
    # This is a placeholder implementation. You can replace it with actual search logic.
    print(f"Searching for: {query}")
    return tavily.search(query=query)


class Source(BaseModel):
    """Schema for a source used by agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent response"""

    answer: str = Field(description="The answer from the agent")
    sources: List[Source] = Field(
        default_factory=list,
        description="A list of sources used to generate the answer",
    )


llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Search for 3 job postings for AI engineer using langchain in linkedin, product company only, in Bangalore or Chennai"
                ),
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
