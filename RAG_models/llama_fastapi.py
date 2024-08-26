import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

# Ensure OPENAI_API_KEY is set correctly


# Set the environment variable for OpenAI API Key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize LLM model
gpt35_llm = OpenAI(model="gpt-3.5-turbo")

# Load documents
documents = SimpleDirectoryReader(r'C:\Users\localadmin\Desktop\myprojects\linkedinsucks\linkedinsucks_2\RAG_models\mydata').load_data()
index = VectorStoreIndex.from_documents(documents)

# Define the prompt template
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in the format requested in the query. "
    "If there are options in the query, answer by choosing one or more options as required. "
    "Keep the answers concise and to the point, do not answer long sentences unless necessary or specified. "
    "Keep the answers concise. Answer in one or two words wherever possible. "
    "Query: {query_str}\n"
    "Answer: "
)

# Initialize query engine
query_engine = index.as_query_engine(llm=gpt35_llm)

# Update prompts
qa_prompt_tmpl = PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

# Test a query
test_query = "what was your major during bachelors"
response = query_engine.query(test_query)

# Convert response to string safely
def response_to_string(response):
    # Check if the response object has the 'response' attribute
    if hasattr(response, 'response'):
        return response.response  # Or another appropriate attribute
    else:
        return str(response)  # Fallback to string conversion

formatted_response = response_to_string(response)
print(formatted_response)

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
        # Perform the query
        response = query_engine.query(query)

        # Convert response to string safely
        formatted_response = response_to_string(response)
        print(formatted_response    )
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
