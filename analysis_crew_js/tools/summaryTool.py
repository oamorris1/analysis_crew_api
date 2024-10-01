import os
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader,  PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.tools import tool
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import re
import tiktoken
import logging
from job_manager import append_event, jobs, jobs_lock, Event
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))

embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002", model="text-embedding-ada-002", chunk_size=10)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
sem_text_splitter = SemanticChunker(
   embeddings, breakpoint_threshold_type="interquartile"
)
deployment_name4 = "gpt-4"
llm_gpt4 = AzureChatOpenAI(deployment_name=deployment_name4, model_name=deployment_name4, temperature=0, streaming=True)
deployment_name4o = "gpt-4o"
llm_gpt4o = AzureChatOpenAI(deployment_name=deployment_name4o, model_name=deployment_name4o, temperature=0, streaming=True)
#path = r'C:\Users\Admin\Desktop\erdcDBFunc\crewAIDocSum\documents'
#path_summary =  r'C:\Users\Admin\Desktop\erdcDBFunc\analysis_crew_js\summaries'
#new_path = r'C:\Users\Admin\Desktop\erdcDBFunc\crewAIDocSum\new_documents'


# summary_root_dir = "/Desktop/crew_docs/summaries/"

# home = str(Path.home())

# current_directory = os.getcwd()
# docs_path = home + "/Desktop/crew_docs/documents"
# if not os.path.exists(docs_path): 
    
#     os.makedirs(docs_path)
# summary_full_path = home + summary_root_dir
# if not os.path.exists(summary_full_path):
#     os.makedirs(summary_full_path)

# summaries_path = summary_full_path + "/summaries.json"



def embedding_cost(chunks):
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in chunks])
    # print(f'Total tokens: {total_tokens}')
    # print(f'Cost in US Dollars: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens
 
prompt_template = """
            
           Throughly read, digest and anaylze the content of the document. 
           Produce a thorough, comprehensive  summary that encapsulates the entire document's main findings, methodology,
           results, and implications of the study. Ensure that the summary is written in a manner that is accessible to a general audience
           while retaining the core insights and nuances of the original paper. Include ALL key terms, definitions, descriptions, points of interest
            statements of facts and concepts, and provide any and all necessary context
           or background information. The summary should serve as a standalone piece that gives readers a comprehensive understanding
           of the paper's significance without needing to read the entire document. Be as THOROUGH and DETAILED as possible.  You MUST
           include all concepts, techniques, variables, studies, research, main findings, key terms and definitions and conclusions. 
           The summary MUST be long enough to capture ALL information in the document:
"{text}"
Thorough SUMMARY:"""

prompt = PromptTemplate.from_template(prompt_template)

class ObtainDocSummary():
   @tool("Document_Summary")
   def doc_sum(docs_path, summaries_path):
        """ Use this tool to access the document folder and summarize a document"""
        
        
        summaries =[]
        
        for file_Name in os.listdir(docs_path):
          text="" 
          full_file_path = os.path.join(docs_path, file_Name)
          print("file name: ", file_Name)
          print("path : ", full_file_path)
          #  code to handle other files than pdf  
          if file_Name.endswith(".txt"):
             loader = TextLoader(full_file_path) 
          elif file_Name.endswith('.pdf'):
            #print("Preparing to sumarize: ", file_Name) 
            #loader = PyPDFLoader(full_file_path)
            loader = PDFMinerLoader(full_file_path)
          elif file_Name.endswith('.docx'):
            loader = Docx2txtLoader(file)
          else:
            print('The document format is not supported!') 
          #loader = PyPDFLoader(full_file_path)
          document = loader.load()
          for page in document:

            text += page.page_content
          #print("Preparing text for: ", file_Name) 
          text = text.replace('\t', ' ')
          text = text.replace('\t', ' ')
          text= text.replace("\n", ' ')
          text = re.sub(" +", " ", text)
          text = re.sub("\u2022", "", text)
          text = re.sub(" +", " ", text)
          text = re.sub(r"\.{3,}", "", text)
          #print("This is the text: ", text, end='\n')
          chunks = text_splitter.create_documents([text])
          sem_chunksCD = sem_text_splitter.create_documents([text])
          #print("Prepared semcd chunks for: ", file_Name) 
          num_tokens = embedding_cost(sem_chunksCD)
          #print("Number of tokens: ", num_tokens, "for: ", file_Name)
          
            
            
          llm_chain = LLMChain(llm=llm_gpt4o, prompt=prompt)
          stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
          
          summary_dict = stuff_chain.invoke(sem_chunksCD)
          #print("chain run completed for: ", file_Name)
            
          summary = summary_dict.get("output_text")
          
               
          new_file_name = file_Name.strip(".pdf")
          summaries.append({"title": file_Name, "summary":summary, "path":full_file_path})
          with open('summaries.json', 'w') as file:
               json.dump(summaries, file)  # Saving the list as JSON
          with open(f'{summaries_path}\{new_file_name}_Summary.txt', "a") as file:
               file.writelines(summary)
              #append these to event log:
          logging.info("Summary written: ", new_file_name)
          logging.info(f"Summary completed for file: {file_Name} ", user="System")
        logging.info("Summarization of all files completed", user="System")
        return summaries