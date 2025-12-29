from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.documents import Document
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# Load environment and setup
def init_environment():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="research/.env")
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load LLM
from langchain_huggingface import HuggingFaceEndpoint

def load_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation", 
        temperature=0.4,
        max_tokens=300
    )



# Load Pinecone retriever

def load_retriever():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("medibot")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    docsearch = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text" 
    )

    return docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})




# Get final answer using both Pinecone & Wikipedia
def query_with_context(query, retriever, llm):
    pinecone_docs = retriever.invoke(query)
    pinecone_context = "\n".join([doc.page_content for doc in pinecone_docs])

    wiki = WikipediaAPIWrapper()
    try:
        wiki_context = wiki.run(query)
    except Exception:
        wiki_context = "Wikipedia info not available."

    combined_doc = Document(
        page_content=f"{pinecone_context}\n\n{wiki_context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful and kind medical assistant. "
            "Use the following retrieved medical context to answer the patient's question. "
            "Explain the answer in simple, easy-to-understand language. "
            "{context}"
        )),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    response = qa_chain.invoke({
        "input": query,
        "context": [combined_doc]
    })

    return response
