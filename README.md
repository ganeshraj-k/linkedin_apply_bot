LinkedIn EasyApply Bot: Automate Your Job Hunt

Overview

This project is designed to automate job applications listed under LinkedIn's EasyApply feature using a combination of Selenium for web automation and an Agentic RAG system for answering questions within the application. The bot can handle repetitive tasks like job searches, applying to positions, and answering application questions, making your job application process more efficient.
Features

    Automation with Selenium: Automates the process of logging into LinkedIn, searching for jobs, and applying to positions listed under EasyApply.
    Query Engine for Application Questions: Uses a combination of hashing, fuzzy matching, and an Agentic RAG system to answer questions automatically.
    Agentic RAG System: Integrates a hybrid model using phi-3-mini for short answer questions and GPT-3.5 for long-form, summary-based responses.

Project Structure

    Automation: Handles job search, applying to jobs, and interaction with web elements using Selenium.
    Querying and Answering: Uses a JSON file to store frequently asked questions. For new or fuzzy matches, an Agentic RAG system is used to query and generate appropriate answers based on the user's resume.

Prerequisites

To run this project, you will need the following:

    Python 3, pip package manager, and Jupyter Notebook.
    Hugging Face and OpenAI API keys (Note: OpenAI requires credits).
    Edge WebDriver or any other popular browser’s WebDriver for Selenium.
    Ollama (for local inference with phi-3-mini). If you're using only GPT-3.5, this step can be skipped.

Setting up Ollama for Local Inference:

    Download Ollama here.
    In a command line, run:
    Ollama run phi3:3.8b-mini-4k-instruct-q4_K_M
    For local inference, use a device with a CUDA-supported GPU for better performance.

Setting up Environment:

    Create a virtual environment to avoid conflicts with your existing Python environment as the project involves several dependencies.

Installation Instructions

    Clone the Project:
    Clone the repository and download the following folders:
        linkedin_apply_bot/bot_python
        linkedin_apply_bot/uvicorn_rag_apis
        linkedin_apply_bot/all_data

    Prepare Data:
        Place your resume and any other relevant profile documents inside all_data/docs.
        Open all_data/search_details/search_details.json and add the roles you're searching for, location, and job search filters.

    Configure Authentication:
        Create a secrets.json file containing your LinkedIn login credentials and OpenAI API key:

        json

    {
      "linkedin_login": "your_login",
      "linkedin_password": "your_password",
      "openai_key": "your_openai_key"
    }

    Create another file titled questions.json with an empty dictionary ({}) for caching purposes.

Create a Log for Applications:

    Create an Excel file titled applied_jobs.xlsx for logging the details of applied jobs.

Modify Paths:

    Update the file paths in bot_sept21.ipynb and the files in the uvicorn_rag_apis folder to reflect your directory structure.

Install Dependencies:

    Download the requirements.txt file and install the necessary packages:

        pip install -r requirements.txt

Running the Bot

    Start the RAG API:
        Navigate to the uvicorn_rag_apis folder in the command line and run the following command to start the API:

        lua

        uvicorn agentic_rag_api:app --reload

        If using only GPT-3.5, replace agentic_rag_api with gpt35_rag_api in the command.

    Run the Bot:
        Open bot_sept21.ipynb in Jupyter Notebook and run the cells. The bot will start applying to jobs automatically.

    Check Logs:
        The job details will be logged in the applied_jobs.xlsx file.

Notes

    Ensure that the paths in the code are properly configured for your system.
    For any new or custom job application questions, the RAG system will generate answers based on your resume.

Project Walkthrough

A detailed walkthrough of the project, including code explanations and a video demo, can be found in this Medium article.
Connect with Me

Feel free to reach out or follow me on LinkedIn: Ganesh's LinkedIn.
