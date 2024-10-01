from flask import Flask, jsonify, request, abort
from uuid import uuid4
from threading import Thread
from datetime import datetime
from flask_cors import CORS
from crewai import Agent, Task, Crew, Process
from crewai.agents import CrewAgentExecutor
from crewai.project import CrewBase, agent, crew, task
from langchain_core.callbacks import BaseCallbackHandler
from typing import TYPE_CHECKING, Any, Dict, Optional
from typing import Union, List, Tuple, Dict
from langchain_openai import AzureChatOpenAI
from langchain_core.agents import AgentFinish
from pathlib import Path

from tools.queryAnalysisTool import QueryDocumentAnalysis
from tools.summaryTool import ObtainDocSummary
from tools.docsynthesisTool import DocumentSynthesisTool
#from tools.docsynthesisToolAzure import AzureDocumentSynthesisTool
from job_manager import append_event, jobs, jobs_lock, Event
import logging
import sys
import threading
import time 
import os
import json

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))

from crew import DocumentSummaryCrew, DocumentAnalysisCrew


load_dotenv(find_dotenv('.env'))

deployment_name3 = "gpt-35-turbo-16k"
deployment_name4 = "gpt-4"
llm_gpt3 = AzureChatOpenAI(deployment_name=deployment_name3, model_name=deployment_name3, temperature=0, streaming=True)
llm_gpt4 = AzureChatOpenAI(deployment_name=deployment_name4, model_name=deployment_name4, temperature=0, streaming=True)
deployment_name4o = "gpt-4o"
llm_gpt4o = AzureChatOpenAI(deployment_name=deployment_name4o, model_name=deployment_name4o, temperature=0, streaming=True)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


summary_root_dir = "/Desktop/crew_docs/summaries/"

home = str(Path.home())

current_directory = os.getcwd()
docs_path = home + "/Desktop/crew_docs/documents"
if not os.path.exists(docs_path): 
    
    os.makedirs(docs_path)
summary_full_path = home + summary_root_dir
if not os.path.exists(summary_full_path):
    os.makedirs(summary_full_path)

summaries_path = summary_full_path + "/summaries.json"





def kickoff_analysis_crew(job_id, user_query=None, docs_path=None, summaries_path=None ):
    logging.info(f"Crew for job {job_id} is starting")

    results = None
    try:
        document_analysis_crew = DocumentAnalysisCrew(job_id)
        document_analysis_crew.setup_crew(user_query, docs_path, summaries_path)
        results = document_analysis_crew.kickoff()
        logging.info(f"Crew for job {job_id} is complete", results)

    except Exception as e:
        logging.error(f"Error in kickoff_crew for job {job_id}: {e}")
        append_event(job_id, f"An error occurred: {e}")
        with jobs_lock:
            jobs[job_id].status = 'ERROR'
            jobs[job_id].result = str(e)

    with jobs_lock:
        jobs[job_id].status = 'COMPLETE'
        jobs[job_id].result = results
        jobs[job_id].events.append(
            Event(timestamp=datetime.now(), data="Crew complete"))




@app.route('/api/crew', methods=['POST'])
def run_crew():
    try:
        logging.info("Received request to run crew")
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data found"}), 400
        if 'user_query' not in data:
            return jsonify({"error": "Invalid input data: 'user_query' missing "}), 400

        #task_type = data['task_type']
        job_id = str(uuid4())
        user_query = data['user_query']
        # docs_path = "C:/Users/Admin/Desktop/erdcDBFunc/analysis_crew_js/documents"
        # summaries_path = "C:/Users/Admin/Desktop/erdcDBFunc/analysis_crew_js/summaries.json"
        
        #if task_type == "document analysis":
            
        thread = Thread(target= kickoff_analysis_crew, kwargs={'job_id': job_id, 'user_query': user_query, 'docs_path': docs_path, 'summaries_path': summaries_path})
        thread.daemon = True
        thread.start()
        return jsonify({'job_id': job_id}), 200
        
    except ValueError:
        # Handle the case where the request body is not valid JSON
        return jsonify({"error": "Invalid JSON format"}), 400

    except Exception as e:
        # Catch-all for any other errors
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An internal error occurred"}), 500
    
@app.route('/api/crew/<job_id>', methods=['GET'])
def get_status(job_id):

    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            abort(404, description="Job not found")

    try:
        result_json = json.loads(job.result)
    except:
        result_json = job.result

    return jsonify({
        "job_id": job_id,
        "status": job.status,
        "results": result_json,
        "events":  [{"timestamp": event.timestamp.isoformat(), "data": event.data} for event in job.events]
    }),200





if __name__ == '__main__':
    app.run(debug=True, port=3001)
