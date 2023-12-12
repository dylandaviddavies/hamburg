from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

class Summarize(BaseModel):
    text: str
    
@app.post('/summarize')
def summarize(summarize: Summarize):
  return summarizer(summarize.text, max_length=300, min_length=30, do_sample=False)