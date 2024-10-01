import os
import json
import re
import time
from typing import Dict, List
import threading
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader, Docx2txtLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from crewai_tools import tool
from langchain.agents import Tool
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.vectorstores.azuresearch import AzureSearch
from panel_interface import chat_interface
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env'))
import panel as pn

embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002", model="text-embedding-ada-002", chunk_size=10)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
semantic_text_splitter = SemanticChunker(embeddings, breakpoint_threshold_amount="interquartile")
deployment_name3 = "gpt-35-turbo-16k"
deployment_name4 = "gpt-4"
llm_gpt3 = AzureChatOpenAI(deployment_name=deployment_name3, model_name=deployment_name3, temperature=0, streaming=True)
llm_gpt4 = AzureChatOpenAI(deployment_name=deployment_name4, model_name=deployment_name4, temperature=0, streaming=True)

sem_text_splitter = SemanticChunker(
    embeddings, breakpoint_threshold_type="interquartile"
)
message_params = dict(
    default_avatars={"User": "👨🏾‍🦱"},
)

# Global variables for callback
user_input = None
initiate_vector_store_task_created = False
use_azure = None

def initiate_vector_store_selection(message):
    global user_input
    global use_azure
    print("At initiate point, made it through callback")
    
    chat_interface.send("Would you like to initiate a call to use Azure for your vector index database? (Yes/No)", user="System", respond=False)
    print("Here is the message from callback: ", message)
    while user_input is None:
        time.sleep(0.1)
    
    use_azure = user_input.lower() in ("y", "yes")
    print("use_azure value is: ", use_azure)
    user_input = None

def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    global initiate_vector_store_task_created
    global user_input
    print("at callback")
    user_input = contents

    if not initiate_vector_store_task_created:
        initiate_vector_store_task_created = True
        thread = threading.Thread(target=initiate_vector_store_selection, args=(contents,))
        thread.start()

def initialize_chat_interface():
    global chat_interface
    if chat_interface is None:
        chat_interface = pn.chat.ChatInterface(message_params=message_params, callback=callback)
        chat_interface.send("Would you like to initiate a call to use Azure for your vector index database? (Yes/No)", user="System", respond=False)

    #return chat_interface

class DocumentSynthesisTool:
    @tool("Document_Synthesis")
    def synthesize_documents(query: str, documents: List[Dict]):
        """
        Synthesizes single or multiple documents by extracting key information, themes, and narratives. 
        Provides a comprehensive and accessible overview of all relevant findings.
        Parameters:
            query (str): The user query to be answered.
            documents (list): A list of dictionaries containing details about the documents to be analyzed.
        Returns:
            str: A comprehensive synthesis of the information relevant to the query.
        """
        print("Made it here inside class declaration1")
        synthesized_information = ""
        
        # Ensure the vector store selection is initiated
        print("state of initiate: ", initiate_vector_store_task_created)
        chat_interface = initialize_chat_interface()
        chat_interface.send("Would you like to initiate a call to use Azure for your vector index database? (Yes/No)", user="System", respond=False)

        print("Point 1 call initiate func")

        while use_azure is None:
            time.sleep(0.1)

        for document in documents:
            title = document['title']
            path = document['path']
            file_name = os.path.basename(path)
            chat_interface.send(f"Preparing: {file_name} for processing", user="System")
            
            try:
                if title.endswith(".txt"):
                    loader = TextLoader(path)
                elif file_name.endswith('.pdf'):
                    loader = PDFMinerLoader(path)
                elif title.endswith('.docx'):
                    loader = Docx2txtLoader(path)
                else:
                    chat_interface.send('The document format is not supported!', user="System", respond=False)
                    continue
                
                document = loader.load()
                text = " ".join(page.page_content for page in document)
                text = re.sub(r"\s+", " ", text)
                text = re.sub(r"\.{3,}", "", text)
                
                sem_chunksCD = sem_text_splitter.create_documents([text])
                print("last use_azure value check, value is: ", use_azure)
                if use_azure:
                    chat_interface.send("Using Azure Search for vectorization and search", user="System", respond=False)
                    index_name = "crewai-vector-index-new1"
                    vector_store = AzureSearch(
                        azure_search_endpoint=os.environ.get("SEARCH_ENDPOINT"),
                        azure_search_key=os.environ.get("SEARCH_API_KEY"),
                        index_name=index_name,
                        embedding_function=embeddings.embed_query,
                    )
                    vector_store = AzureSearch.add_documents(documents=sem_chunksCD)
                else:
                    chat_interface.send("Using FAISS for vectorization and search", user="System", respond=False)
                    vector_store = FAISS.from_documents(sem_chunksCD, embeddings)
                
                retriever = vector_store.as_retriever()
                qa_chain = RetrievalQA.from_chain_type(llm=llm_gpt4, retriever=retriever)
                
                result = qa_chain.invoke({"query": query})
                synthesized_information += f"Title: {title}\n{result['result']}\n\n"
            
            except Exception as e:
                chat_interface.send(f"An error occurred while processing document {title}: {e}", user="System", respond=False)
        
        return synthesized_information

