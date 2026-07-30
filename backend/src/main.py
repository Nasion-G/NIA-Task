from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torchaudio
import subprocess
import os
import codec
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

OUTPUT_DIR = "src/files/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_unique_path(directory: str, base_name: str, ext: str) -> str:
    path = os.path.join(directory, f"{base_name}{ext}")
    if not os.path.exists(path):
        return path

    i = 1
    while True:
        candidate = os.path.join(directory, f"{base_name}-{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1

class EncodedInput(BaseModel):
    codes: list[list[list[int]]]
    filename: str

@app.post("/uploadaudio/")
async def uploadAudio(data: EncodedInput):
    codes = torch.tensor(data.codes)
    wav = codec.decodeAudio(codes)

    base_name = os.path.splitext(data.filename)[0]

    raw_path = get_unique_path(OUTPUT_DIR, f"{base_name}_raw", ".wav")
    output_path = get_unique_path(OUTPUT_DIR, f"{base_name}_final", ".wav")

    torchaudio.save(raw_path, wav.squeeze(0), codec.model.sample_rate)
    subprocess.run(["ffmpeg", "-i", raw_path, "-ar", "22050", "-y", output_path])

    return {"filename": output_path}