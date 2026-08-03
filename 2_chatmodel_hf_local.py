from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# Create FastAPI App
app = FastAPI()

# Home Endpoint
@app.get("/")
def home():
    return {
        "message": "This is a simple chatbot using LangChain and Hugging Face",
        "title": "Implemented by MOHD DANISH"
    }

# Hugging Face Cache
os.environ["HF_HOME"] = "D:/huggingface_cache"

# Load Model
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100
    }
)

# Create Chat Model
model = ChatHuggingFace(llm=llm)

# Request Body
class SimpleChat(BaseModel):
    question: str

# Chat Endpoint
@app.post("/simplechat")
def chat_model(data: SimpleChat):

    response = model.invoke(data.question)

    return {
        "question": data.question,
        "answer": response.content
    }
