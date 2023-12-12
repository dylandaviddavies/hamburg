from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class Summarize(BaseModel):
    text: str
    
@app.post('/summarize')
def summarize(summarize: Summarize):
  return summarizer(summarize.text, max_length=300, min_length=30, do_sample=False)