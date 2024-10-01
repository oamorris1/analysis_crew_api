import os
import json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
import json
import os
import tiktoken
from transformers import AutoTokenizer
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
deployment_name4 = "gpt-4"
docs_path = "C:/Users/Admin/Desktop/erdcDBFunc/analysis_crew/documents"
llm_gpt4 = AzureChatOpenAI(deployment_name=deployment_name4, model_name=deployment_name4, temperature=0, streaming=True)

path_summary =  r'C:\Users\Admin\Desktop\erdcDBFunc\analysis_crew\summaries'


def embedding_cost(chunks):
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in chunks])
    print(f'Total tokens: {total_tokens}')
    print(f'Cost in US Dollars: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens
def token_count(chunks):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    tokens = tokenizer.tokenize(chunks)
    num_tokens = len(tokens)
    print(f"Number of tokens in your text: {num_tokens}")
                 
                 
                 
# cost_tokens = embedding_cost(chunks)
# print(cost_tokens)
# [4/11 12:14 PM] Omar Morris
# that's when you create your chunks like: 
# text_splitter = RecursiveTextSplitter(chunk_size=1000, chunk_overlap=50)
# chunks = text_splitter.split_documents(documents)
# [4/11 12:15 PM] Omar Morris
# documents = pdfyloader(pdf)
# text_splitter = RecursiveTextSplitter(chunk_size=1000, chunk_overlap=50)
# chunks = text_splitter.split_documents(documents)


map_prompt_template = """Throughly read and anaylze the following chunk of  {text}. Based on this please provide a comprehensive summary of the given text. The summary should cover all the key points
           and main ideas presented in the original text, while also condensing the information into a concise and easy-to-understand format.
           The goal is to please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary 
           information or repetition. The length of the summary should be appropriate for the length and complexity of the original text,
           providing a clear and accurate overview without omitting any important information 

"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables = ["text"])
map_chain = LLMChain(llm=llm_gpt4, prompt=map_prompt)
# Reduce
reduce_template = """The following is set of summaries:
{text}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""
combine_prompt_template = """The following is set of summaries:
{text}
Take these and distill it into a final, consolidated summary of the main themes. 
Consolidated Summary:
"""

combine_prompt = PromptTemplate( template= combine_prompt_template, input_variables=["text"])
# reduce_prompt = PromptTemplate.from_template(reduce_template)
# reduce_chain = LLMChain(llm=llm_gpt4, prompt=reduce_prompt)
# combine_documents_chain = StuffDocumentsChain(
#     llm_chain=reduce_chain, document_variable_name="docs"
# )

# Combines and iteratively reduces the mapped documents
# reduce_documents_chain = ReduceDocumentsChain(
#     # This is final chain that is called.
#     combine_documents_chain=combine_documents_chain,
#     # If documents exceed context for `StuffDocumentsChain`
#     collapse_documents_chain=combine_documents_chain,
#     # The maximum number of tokens to group documents into.
#     token_max=4000,
# )



def doc_sum(docs_path):
    text=""   
    textLS="" 
    summaries =[]
    for file_Name in os.listdir(docs_path):
        full_file_path = os.path.join(docs_path, file_Name)
                
        loader = PyPDFLoader(full_file_path)
        print("path: ", full_file_path)
        documentLS = loader.load_and_split()
        document = loader.load()
        for page in document:

            text += page.page_content
    
        text = text.replace('\t', ' ')
        for page in documentLS:
            textLS += page.page_content
    
        textLS = text.replace('\t', ' ')
        pages = loader.load_and_split()
        pdf_pages_content = '\n'.join(page.page_content for page in pages)    
        # https://www.reddit.com/r/LangChain/comments/170mfkc/recursivecharactertextsplitter_create_documents/  
        # create_documents takes an array of strings, splits them and returns documents for them. 
        # split_documents takes an array of documents, turns them into an array of strings a hands them off to create_documents to split and make new documents out of . 
        chunks = text_splitter.split_documents(document)
        chunksCD = text_splitter.create_documents([text])
        chunksCDLS = text_splitter.create_documents([textLS])
       
        #chunks2 = text_splitter.create_documents(document)
        num_tokens = embedding_cost(chunks)
        num_tokensCD = embedding_cost(chunksCD)
        num_tokensCDLS = embedding_cost(chunksCDLS)
        print("Pages is type: ", type(pdf_pages_content))
        print("document is type: ", type(document))
        print("document load and splt is type: ", type(documentLS))
        print("text extract is type: ", type(text))
        print("text Ld and Splt conversion is type: ", type(textLS))
        print("chunks split doc  is type: ", type(chunks))
        print("chunks Create document  is type: ", type(chunksCD))
        print("chunksCD Ld and Splt is type: ", type(chunksCDLS))
        print("document length is: ", len(document))
        print("documentLS length is: ", len(documentLS))
        print("text length is: ", len(text))
        print("textLS length is: ", len(textLS))
        print("chunks length is: ", len(chunks))
        print("chunksCD length is: ", len(chunksCD))
        print("chunksCDLS length is: ", len(chunksCDLS))
        num2 = llm_gpt4.get_num_tokens(text)
        num3 = llm_gpt4.get_num_tokens(textLS)
        #num_tokens2 = embedding_cost(chunks2)
        print("# number of tokens embedding cost func: ", num_tokens)
        print("# number of tokensCD embedding cost func: ", num_tokensCD)
        print("# number of tokensCDLS embedding cost func: ", num_tokensCDLS)
        print("# number of tokens text string from load() and langchin get_num_tok method : ", num2)
        print("# number of tokens text load_and_split()  and  langchin get_num_tok method : ", num3)
        #print(text)
        
        if num_tokens < 135000:
            print("Tokens: ", num_tokens)
            print("running stuff chain")
            chain = load_summarize_chain(llm_gpt4, chain_type="stuff")
            summary_dict = chain.invoke(document)
            
            summary = summary_dict.get("output_text")
               
            new_file_name = file_Name.strip(".pdf")
            summaries.append({"title": file_Name, "summary":summary, "path":full_file_path})
            with open('summaries.json', 'w') as file:
                json.dump(summaries, file)  # Saving the list as JSON
            with open(f'{path_summary}\{new_file_name}_Summary.txt', "w") as file:
                file.writelines(summary)
        # else:
        #     print("Tokens more than stuff can handle: ", num_tokens)
        #     chain = MapReduceDocumentsChain(
        #     # Map chain
        #     llm_chain=map_chain,
        #     # Reduce chain
        #     reduce_documents_chain=reduce_documents_chain,
        #     # The variable name in the llm_chain to put the documents in
        #     document_variable_name="docs",
        #     # Return the results of the map steps in the output
        #     return_intermediate_steps=True,
        # )
            
        else:
            print("Tokens more than stuff can handle: ", num_tokens)
            chain = load_summarize_chain(llm_gpt4,
            # Map chain
            chain_type="map_reduce",
            map_prompt = map_prompt,
            combine_prompt= combine_prompt,
            
            return_intermediate_steps=True,
        )
            print("made it here_1")
            #summary_dict = chain.invoke(chunks)
            summary_dict = chain.invoke({"input_documents":chunks})
            print("made it here_2")
            
            print("made it to this point")
            summary = summary_dict.get("output_text")
                
            new_file_name = file_Name.strip(".pdf")
            summaries.append({"title": file_Name, "summary":summary, "path":full_file_path})
            with open('summaries.json', 'w') as file:
                json.dump(summaries, file)  # Saving the list as JSON
            with open(f'{path_summary}\{new_file_name}_Summary.txt', "w") as file:
                file.writelines(summary)




        # print("made it here")
        # summary_dict = chain.invoke(document)
        # print("made it to this point")
        # summary = summary_dict.get("output_text")
               
        # new_file_name = file_Name.strip(".pdf")
        # summaries.append({"title": file_Name, "summary":summary, "path":full_file_path})
        # with open('summaries.json', 'w') as file:
        #     json.dump(summaries, file)  # Saving the list as JSON
        # with open(f'{path_summary}\{new_file_name}_Summary.txt', "w") as file:
        #     file.writelines(summary)
        
    return summaries


#tokensText = llm_gpt4.get_num_tokens()
test = doc_sum(docs_path)
print(test)