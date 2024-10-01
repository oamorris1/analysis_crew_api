from job_manager import append_event
import logging
from agents import DocumentSummarizeAgents
from tasks import AnalyzeDocumentsTasks
from crewai import Crew
from langchain_openai import AzureChatOpenAI


deployment_name4o = "gpt-4o"
llm_gpt4o = AzureChatOpenAI(deployment_name=deployment_name4o, model_name=deployment_name4o, temperature=0, streaming=True)

class DocumentAnalysisCrew:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.crew = None

    def setup_crew(self,user_query, docs_path, summaries_path):
        logging.info(f"Setting up crew for {self.job_id} for document analysis")


        #agents
        agents = DocumentSummarizeAgents()
        summarizer_agent = agents.document_summary_agent()
        analyzer_agent   = agents.query_analysis_agent()
        docs_analyzer_agent = agents.document_analysis_agent() 

        #tasks
        tasks = AnalyzeDocumentsTasks(job_id=self.job_id)
        doc_sum_task = tasks.summarize_document(summarizer_agent, docs_path)
        analyze_query_task = tasks.analyze_document_query(analyzer_agent, summaries_path, user_query )
        docs_synthesizer_task = tasks.document_sythesis(docs_analyzer_agent, user_query)

        #create crew
        self.crew = Crew(
            agents=[summarizer_agent, analyzer_agent, docs_analyzer_agent],
            tasks=[doc_sum_task, analyze_query_task, docs_synthesizer_task],
            verbose=2,
            manager_llm=llm_gpt4o
        )

    def kickoff(self):
        #kick off crew
        
        if not self.crew:
            logging.info(f"No crew found for {self.job_id}")
            append_event(self.job_id, "Crew not set up")
            return
        
        append_event(self.job_id, "CREW STARTED")
        try: 
            logging.info(f"Running crew for job id:  {self.job_id}")
            results = self.crew.kickoff()
            append_event(self.job_id, "CREW TASKS COMPLETED")
            return results
        
        except Exception as e:
            append_event(self.job_id, f"An error occurred: {e}")
            return str(e)

class DocumentSummaryCrew:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.crew = None

    def setup_crew(self,user_query, docs_path, summaries_path):
        logging.info(f"Setting up crew for {self.job_id} for document summarization")


        #agents
        agents = DocumentSummarizeAgents()
        summarizer_agent = agents.document_summary_agent()
      

        #tasks
        tasks = AnalyzeDocumentsTasks(job_id=self.job_id)
        doc_sum_task = tasks.summarize_document(summarizer_agent, docs_path)
       

        #create crew
        self.crew = Crew(
            agents=[summarizer_agent],
            tasks=[doc_sum_task],
            verbose=2,
        )

    def kickoff(self):
        #kick off crew
        
        if not self.crew:
            logging.info(f"No crew found for {self.job_id}")
            append_event(self.job_id, "Crew not set up")
            return
        
        append_event(self.job_id, "CREW STARTED")
        try: 
            logging.info(f"Running crew for job id:  {self.job_id}")
            results = self.crew.kickoff()
            append_event(self.job_id, "CREW TASKS COMPLETED")
            return results
        
        except Exception as e:
            append_event(self.job_id, f"An error occurred: {e}")
            return str(e)

