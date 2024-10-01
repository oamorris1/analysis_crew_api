# analysis_crew_api
Document Analysis Application
Overview
The Document Analysis Application is a powerful tool designed for document summarization and deep qualitative comparison analysis. It leverages the capabilities of LangChain and the agent framework CrewAI to provide comprehensive and accurate document insights. The application uses agents to perform specific tasks, such as summarizing documents, analyzing user queries, and synthesizing information from multiple documents.

Features
Document Summarization: Provides detailed summaries of documents for better understanding.
Query Analysis: Allows users to input specific queries for analyzing documents.
Synthesis of Multiple Documents: Generates insights by synthesizing information from different documents.
Technologies Used
Flask: For building the REST API.
CrewAI: To manage agent-based tasks and workflows.
LangChain: To interact with Azure OpenAI models for NLP tasks.
Azure OpenAI: Used to handle the language models (gpt-4, gpt-35-turbo-16k).
Threading: For running jobs in the background.
Dotenv: To load configuration from environment variables.
Installation
Prerequisites
Python 3.8 or higher
Virtual Environment (venv)
Azure OpenAI account
Steps to Set Up
Clone the repository:

bash
Copy code
git clone <repository_url>
cd document-analysis-application
Create and Activate a Virtual Environment:

bash
Copy code
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
Install the Required Dependencies:

bash
Copy code
pip install -r requirements.txt
Environment Setup:

Copy the sample environment file:
bash
Copy code
cp .env.sample .env
Edit .env to add your Azure OpenAI credentials:
text
Copy code
AZURE_OPENAI_ENDPOINT="https://your-openai-endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-api-key"
OPENAI_API_VERSION="api-version"
openai.api_type="azure"
Running the Application
To start the Flask application, run:

bash
Copy code
python api.py
The application will run locally on port 3001.

API Endpoints
1. /api/crew [POST]
Kicks off a document analysis process.

Request Body (JSON):
json
Copy code
{
  "user_query": "Your query about the documents"
}
Response:
Returns a job ID for tracking the analysis.
Example:
json
Copy code
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
}
2. /api/crew/<job_id> [GET]
Gets the status and results of a running or completed job.

Path Parameter:
job_id: The ID of the job you wish to check the status of.
Response:
Returns the current status (RUNNING, COMPLETE, ERROR) and, if complete, the results.
Example:
json
Copy code
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "COMPLETE",
  "results": {
    "summary": "Summary of the document...",
    "analysis": "Detailed analysis..."
  },
  "events": [
    {
      "timestamp": "2024-10-01T12:34:56.789Z",
      "data": "Crew complete"
    }
  ]
}
Environment Variables
The .env file must contain the following keys:

AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint.
AZURE_OPENAI_API_KEY: The API key for Azure OpenAI.
OPENAI_API_VERSION: The version of the Azure OpenAI API you are using.
openai.api_type: Should be set to "azure".
A sample .env file is included in the repository for reference.

Example Usage
Below is an example of how to use the endpoints:

Kick Off a Crew:

bash
Copy code
curl -X POST http://localhost:3001/api/crew -H "Content-Type: application/json" -d '{"user_query": "Summarize the document"}'
Response:

json
Copy code
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
}
Check Job Status:

bash
Copy code
curl http://localhost:3001/api/crew/f47ac10b-58cc-4372-a567-0e02b2c3d479
Response:

json
Copy code
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "RUNNING",
  "results": null,
  "events": []
}
Contributing
We welcome contributions! Please fork the repository and create a pull request to propose changes. For major changes, please open an issue to discuss what you would like to change.

License
This project is licensed under the MIT License.

Known Issues and Future Improvements
Currently, only text-based documents are supported for analysis.
Future releases will support additional formats such as PDF and Word documents.
Improvements planned for agent workflows to add more complex analysis features.
