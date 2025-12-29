from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class IngestRequest(BaseModel):
    document_id: str
    blob_url: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    # For now, just echo back; we'll add PDF download later
    return {
        "message": "Ingest request received",
        "document_id": req.document_id,
        "blob_url": req.blob_url,
    }

