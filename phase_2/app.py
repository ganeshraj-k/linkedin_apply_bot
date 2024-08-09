from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAi
from langserve import add_routes

import uvicorn
import os

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


app = FastAPI(
    



)