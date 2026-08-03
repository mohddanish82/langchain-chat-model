from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
@app.get("/")
def home():
    return {
    "message": "This is a simple chatbot using LangChain and Hugging Face",
    "title": "Implemented by MOHD DANISH"
    }
os.environ['HF_HOME'] = 'D:/huggingface_cache'
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

class Simplechat(BaseModel):
    question: str


@app.post("/simplechat")
def chat_model(data: Simplechat):
    respone = model.invoke(data.question)
    return{
        "question": data.question,
        "answer": response.content
    }
