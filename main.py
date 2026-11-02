import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initialising components..")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retreiver = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context: 

    {context}

    Question: {question}

    Provide a detailed answer:
    """)


def format_docs(docs):
    """Format retreived vector documents into a single string

    Args:
        docs (_type_): retreived vector documents
    """
    return "\n\n".join(doc.page_content for doc in docs)


def retreival_chain_without_lcel(query: str):
    """Simple retreival chain without LCEL

    Args:
        query (str): query to search for in vector store
    """

    docs = retreiver.invoke(query)
    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)

    return response.content[0]["text"]


def create_retreival_chain_with_lcel():
    retreival_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retreiver | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retreival_chain


if __name__ == "__main__":
    print("Retreieving...")

    query = "what is pinecone in Machine learning?"

    print("\n" + "-" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation without RAG")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content[0]["text"])

    print("\n" + "-" * 70)
    print("IMPLEMENTATION 1: RAG Without LCEL")
    print("=" * 70)
    rag_result_without_lcel = retreival_chain_without_lcel(query)
    print("\nAnswer:")
    print(rag_result_without_lcel)

    print("\n" + "-" * 70)
    print("IMPLEMENTATION 2: Raw LLM Invocation without RAG")
    print("=" * 70)
    chain_with_lcel = create_retreival_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer: ")
    print(result_with_lcel)
