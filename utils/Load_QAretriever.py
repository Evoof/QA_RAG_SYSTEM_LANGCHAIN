from warnings import filterwarnings
filterwarnings(action="ignore")
from ProjectConfiguration.SecretKeys import ProjectConfig

import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["GROQ_API_KEY"] = ProjectConfig.groq_api_key

web_path = "https://medium.com/@spaw.co/best-website|s-to-practice-web-scraping-9df5d4df4d1"

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

def LoadRetriever(web_path:str, llm):
    webloader = WebBaseLoader(
                          web_path=web_path,
                          # bs_kwargs={"parse_only":bs4.SoupStrainer(["p", "h1"])}
                         )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=100,
                                                   length_function=len)
    # Step 1: Load the document
    loaded_doc = webloader.load()
    
    # Step 2: Split the document
    splitted_doc = text_splitter.split_documents(loaded_doc)
    
    print(f"Document splitted into: {len(splitted_doc)} document")
    
    # Step 3: Text Embedding 
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    
    # Step 4: Embed and Store Text
    vector_embeddings = FAISS.from_documents(documents=splitted_doc, embedding=embedding)
    
    # Step 5: Create retriever chain
    vectore_retriver = vector_embeddings.as_retriever(search_type="similarity",
                                                      search_kwargs={"k":5})
    
    print("Retriever created")

    prompt = ChatPromptTemplate.from_messages([
        ("system", ("You are a helpful and concise AI assistant." 
                    "Use the provided context to answer the user's question." 
                    "If the answer is not in the context, respond with 'I don't know'.")),
        ("user", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    qa_retriever = RetrievalQA.from_chain_type(llm=llm,
                                               retriever=vectore_retriver,
                                               chain_type="stuff", 
                                               chain_type_kwargs={"prompt":prompt})

    return qa_retriever


retriever = LoadRetriever(web_path=web_path, llm=llm)