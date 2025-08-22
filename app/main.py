from fastapi import FastAPI
from pydantic import BaseModel
from .nlp import similarity_score

app = FastAPI(title="ATS Buddy")

class CompareReq(BaseModel):
    resume_text: str
    job_text: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/compare")
def compare(req: CompareReq):
    return similarity_score(req.resume_text, req.job_text)