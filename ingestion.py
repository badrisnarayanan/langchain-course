import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = UnstructuredLoader(
        file_path="/Users/badrinarayanans/Desktop/learning/eden_marco_langchain/langchain-course/mediumblog1.txt",
        chunking_strategy="basic",
        max_characters=10000000,
    )
    document = loader.load()

    print("splitting..")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    print("ingesting..")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
