import os
import re
import time
import threading
from typing import Dict, List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from crewai_tools import tool
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from panel_interface import chat_interface
from dotenv import load_dotenv, find_dotenv
import panel as pn

# Load environment variables
load_dotenv(find_dotenv('.env'))

# Initialize embeddings and LLMs
embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002", model="text-embedding-ada-002", chunk_size=10)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
semantic_text_splitter = SemanticChunker(embeddings, breakpoint_threshold_amount="interquartile")

deployment_name3 = "gpt-35-turbo-16k"
deployment_name4 = "gpt-4"
deployment_name4o = "gpt-4o"

llm_gpt3 = AzureChatOpenAI(deployment_name=deployment_name3, model_name=deployment_name3, temperature=0, streaming=True)
llm_gpt4 = AzureChatOpenAI(deployment_name=deployment_name4, model_name=deployment_name4, temperature=0, streaming=True)
llm_gpt4o = AzureChatOpenAI(deployment_name=deployment_name4o, model_name=deployment_name4o, temperature=0, streaming=True)

# Global variables
user_input = None
initiate_chat_task_created = False

def custom_ask_human_input(prompt: str) -> str:
    global user_input

    chat_interface.send(prompt, user="System", respond=False)

    while user_input is None:
        time.sleep(1)

    human_response = user_input
    user_input = None

    return human_response

def initiate_chat(message):
    global initiate_chat_task_created
    initiate_chat_task_created = True
    custom_ask_human_input(message)

def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    global initiate_chat_task_created
    global user_input

    if not initiate_chat_task_created:
        thread = threading.Thread(target=initiate_chat, args=(contents,))
        thread.start()
    else:
        user_input = contents

def initialize_chat_interface():
    global chat_interface
    print("user input value is: ", user_input)
    
    chat_interface = pn.chat.ChatInterface(callback=callback)
        #user_response = custom_ask_human_input("Would you like to use Azure for your vector index database? Any response other than Yes or Y (not case sensitive) will be treated as No.")
    chat_interface.send("Use azure?", respond=False)
    while user_input is None:
        time.sleep(1)

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
        synthesized_information = ""
        for document in documents:
            title = document['title']  # Access the title of the document
            path = document['path']  # Document path
            file_name = os.path.basename(path)
            chat_interface.send(f"Preparing: {file_name} for processing", user="System")
            try:
                if title.endswith(".txt"):
                    loader = TextLoader(path)
                    chat_interface.send("Processing : ", file_name)  
                elif file_name.endswith('.pdf'):
                    chat_interface.send(f"Processing: {file_name} ", user="Assistant") 
                    loader = PDFMinerLoader(path)
                elif title.endswith('.docx'):
                    loader = Docx2txtLoader(path)
                    chat_interface.send("Processing: ", file_name) 
                else:
                    chat_interface.send('The document format is not supported!', user="System") 
                    continue

                document = loader.load()
                text = ""
                for page in document:
                    text += page.page_content
                text = text.replace('\t', ' ')
                text = text.replace('\n', ' ')
                text = re.sub(" +", " ", text)
                text = re.sub("\u2022", "", text)
                text = re.sub(r"\.{3,}", "", text)

                sem_chunksCD = semantic_text_splitter.create_documents([text])
                
                # Ask human input for vectorization choice
                print("here")
                user_response=initialize_chat_interface()
                #user_response = custom_ask_human_input("Would you like to use Azure for your vector index database? Any response other than Yes or Y (not case sensitive) will be treated as No.")
                while user_input is None:
                    time.sleep(1)
                if user_response.lower() not in ("y", "yes"):
                    chat_interface.send("We will use FAISS for vectorization and search", user="System")
                    vector_store = FAISS.from_documents(sem_chunksCD, embeddings)
                else:
                    chat_interface.send("We will use Azure Search for vectorization and search", user="System")
                    # Assuming AzureSearch setup here (needs proper implementation)
                    #vector_store = AzureSearch.from_documents(sem_chunksCD, embeddings)

                retriever = vector_store.as_retriever()
                qa_chain = RetrievalQA.from_chain_type(llm=llm_gpt4o, retriever=retriever)
                result = qa_chain.invoke({"query": query})
                synthesized_information += f"Title: {title}\n{result['result']}\n\n"

            except Exception as e:
                chat_interface.send(f"An error occurred while processing document {title}: {e}", user="System")

        return synthesized_information

