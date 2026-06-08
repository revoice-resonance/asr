import torch
import io
import time
import tempfile
import subprocess
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

app = FastAPI(title="Whisper Finetuned API")

MODEL_PATH = "/root/autodl-tmp/models/whisper-large-v3-turbo-finetuned"

print("Loading model...")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
model.to("cuda")
model.eval()
print(f"Model loaded on CUDA. VRAM used: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device="cuda",
)

def load_audio_ffmpeg(audio_bytes: bytes, sr: int = 16000) -> np.ndarray:
    """Use ffmpeg to decode any audio format to numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        cmd = [
            "ffmpeg", "-i", tmp.name,
            "-f", "f32le", "-ac", "1", "-ar", str(sr),
            "-hide_banner", "-loglevel", "error",
            "pipe:1"
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")
        audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio

@app.get("/health")
def health():
    return {"status": "ok", "model": "whisper-large-v3-turbo-finetuned", "device": "cuda"}

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default="zh"),
    response_format: str = Form(default="json"),
):
    start = time.time()
    audio_bytes = await file.read()
    audio_array = load_audio_ffmpeg(audio_bytes, sr=16000)

    result = pipe(
        audio_array,
        generate_kwargs={"language": language, "task": "transcribe"},
        return_timestamps=True,
    )

    elapsed = time.time() - start

    if response_format == "verbose_json":
        return JSONResponse({
            "text": result["text"],
            "chunks": result.get("chunks", []),
            "duration": float(len(audio_array) / 16000),
            "processing_time": round(elapsed, 3),
        })
    return JSONResponse({"text": result["text"]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
