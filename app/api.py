from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.scoring import analyze, extract_urls
from app.privacy import mask_pii
from app.threat_intel import check_feeds

app = FastAPI(title="CyberShield API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Email, SMS or URL to check")


class AnalyzeResponse(BaseModel):
    risk: str
    score: float
    reasons: list[str]
    critical_rules: list[str]
    urls: list[dict]
    text_score: float
    feed_override: bool
    pii_masked: dict[str, int]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_message(req: AnalyzeRequest):
    # Privacy: personal data is masked before scoring, and the raw text is not stored
    masked, counts = mask_pii(req.text)
    hits = check_feeds(extract_urls(masked))
    result = analyze(masked, feed_hits=hits)
    result["pii_masked"] = counts
    return result