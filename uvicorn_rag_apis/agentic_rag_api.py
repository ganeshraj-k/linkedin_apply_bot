import os
import json
import torch
import uvicorn
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, Settings, load_index_from_storage, StorageContext
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Paths
secrets_path = r"C:\Users\localadmin\Desktop\new desktop\linkedinsucks\user_files\secrets.json"
persistent_storage_path = r"C:\Users\localadmin\Desktop\new desktop\linkedinsucks\linkedin_easyapply_bot\all_data\persistent_storage_index"
docs_path = r"C:\Users\localadmin\Desktop\new desktop\linkedinsucks\linkedin_easyapply_bot\all_data\docs"

PERSIST_DIR = r"C:\Users\localadmin\Desktop\new desktop\linkedinsucks\linkedin_easyapply_bot\all_data\persistent_storage"
vector_index_dir = PERSIST_DIR + r"\vector"
summary_index_dir = PERSIST_DIR + r"\summary"

# Check if CUDA is available
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Using GPU: {gpu_name}")
else:
    print("CUDA is not available. Using CPU.")

# Load secrets
with open(secrets_path, 'r') as file:
    data = json.load(file)

OPENAI_API_KEY = data["OPENAI_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Function to check if persistent storage exists
def check_persist(storage_path):
    return os.path.exists(storage_path)

# Initialize LLMs and embeddings
gpt3_5_llm = OpenAI(model="gpt-3.5-turbo")
gpt3_5_embed = OpenAIEmbedding(model="text-embedding-ada-002")

system_prompt = """
answer questions about a candidate's resume and profile
"""

phi3_llm = Ollama(
    model="phi3:3.8b-mini-4k-instruct-q4_K_M",
    temperature=0.01,
    request_timeout=400,
    system_prompt=system_prompt,
    context_window=2000
)

phi3_embed = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# QA Prompt templates
qa_prompt_tmpl_str_gpt4 = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge,
it is very important that you follow the instructions clearly and answer based on the given information.
Answer the query in the format requested in the query.
Do not leave blanks in the answers.
Always answer the questions in the form of a cover letter and in first person.
If asked for a cover letter, write a short cover letter talking about your previous work experience and how it would make you a good fit for the given role.
If asked about why you want to work at a certain company, write a concise cover letter including the company's name and talking about your previous work experience and how it would make you a good fit for the given role.

Query: {query_str}
Answer: \
"""

qa_prompt_tmpl_str_phi3 = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge,
Answer the query in the format requested in the query.
It is very important and a priority to keep the answers extremely short and concise.
Keep the answers concise and as short as possible. Answer in one or two words wherever possible. Keep the answers short, do not elaborate unless necessary, do not explain or elaborate.
If there are options in the query, answer by choosing one or more options as required.
When asked for city, return the city name along with the state.
Return only one answer, do not return multiple answers or list of answers unless specified.
For queries which ask for years of experience, always return values greater than 1.
For queries like "how many years of experience do you have with some tool", return just the integer.
For queries like "how many years of experience", the answer should always be an integer.
For questions that start like "do you have experience with", always return "Yes".
For queries that begin with "Experience with", they are asking the number of years of experience; treat this query the same as those that ask for the number of years of experience with a certain tool.
For queries that ask "are you willing to relocate" or "are you local to a certain place", always answer "Yes".
Keep the answers concise and to the point, do not answer long sentences unless necessary or specified.
Do not explain the answer or do not generate any text in addition to the answer. 


Query: {query_str}
Answer: \
"""

# Function to get the summarization tool using GPT-3.5
def get_summary_tool(docs):
    splitter = SentenceSplitter(chunk_size=1024)
    documents = splitter.get_nodes_from_documents(docs)

    Settings.llm = gpt3_5_llm
    Settings.embed_model = gpt3_5_embed

    if check_persist(summary_index_dir):
        storage_context = StorageContext.from_defaults(persist_dir=summary_index_dir)
        summary_index = load_index_from_storage(storage_context)
    else:
        summary_index = VectorStoreIndex(documents)
        summary_index.storage_context.persist(summary_index_dir)

    summary_query_engine = summary_index.as_query_engine(response_mode="refine", use_async=True)
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str_gpt4)
    summary_query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

    summary_tool = QueryEngineTool.from_defaults(
        name="summary_tool",
        query_engine=summary_query_engine,
        description="Use only for long answer and summary based questions about the profile. do not use otherwise or if not necessary"
    )

    return summary_tool

# Function to get the vector search tool using phi3-mini
def get_vector_tool(docs):
    # splitter = SentenceSplitter(chunk_size=1024)
    # documents = splitter.get_nodes_from_documents(docs)

    Settings.llm = phi3_llm
    Settings.embed_model = phi3_embed

    if check_persist(vector_index_dir):
        storage_context = StorageContext.from_defaults(persist_dir=vector_index_dir)
        vector_index = load_index_from_storage(storage_context)
    else:
        vector_index = VectorStoreIndex(docs)
        vector_index.storage_context.persist(vector_index_dir)

    vector_query_engine = vector_index.as_query_engine(response_mode='compact', use_async=True)
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str_phi3)
    vector_query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

    vector_tool = QueryEngineTool.from_defaults(
        name="vector_tool",
        query_engine=vector_query_engine,
        description="Useful for short questions about the profile.Use this tool always until specificall requested for long form answers or if long form answers are required. Prioritize this tool"
    )

    return vector_tool

# Example usage
docs = SimpleDirectoryReader(docs_path).load_data()

# Get the summarization and vector tools
summary_tool = get_summary_tool(docs)
vector_tool = get_vector_tool(docs)

# Now you can use summary_tool and vector_tool for your queries
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    
)


# Initialize FastAPI
app = FastAPI()

# Define request and response models using Pydantic
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# Define the endpoint for querying the model
@app.post("/resume_qa", response_model=QueryResponse)
async def generate_text(request: QueryRequest):
    query = request.query
    try:
        response = query_engine.query(query)
        formatted_response = str(response.response)
        print(formatted_response)
        return QueryResponse(response=formatted_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
