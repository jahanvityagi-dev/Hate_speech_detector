from fastapi import FastAPI, HTTPException,UploadFile,File
from pydantic import BaseModel
from app.services.moderation_pipeline import ModerationPipeline
from app.schemas.moderation import ModerationRequest, ModerationResponse
from app.services.audio_transcript_service import AudioTranscriptionService
import os
import whisper
from app.agents.error_handler_agent import ErrorHandlerAgent

import tempfile
import logging
logging.basicConfig(level=logging.INFO)


transcriber = AudioTranscriptionService()
app = FastAPI(
    title="Hate Speech Detection API",
    description="Detects and explains hate speech, retrieves relevant policies, and suggests moderation actions.",
    version="1.0.0"
)
model = whisper.load_model("base")
error_handler = ErrorHandlerAgent()

pipeline = ModerationPipeline()

class InputText(BaseModel):
    text: str


@app.post("/moderate", response_model=ModerationResponse)
def moderate(input: ModerationRequest):
    try:
        result = pipeline.run(input.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    async def process():
        suffix = os.path.splitext(file.filename)[-1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        result = model.transcribe(tmp_path)
        text = result["text"]
        moderation_result = pipeline.run(text)
        os.remove(tmp_path)

        return {
            "transcribed_text": text,
            "moderation_result": moderation_result
        }

    return await error_handler.safe_execute_async(process)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Backend is running"}

