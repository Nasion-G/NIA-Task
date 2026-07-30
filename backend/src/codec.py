from encodec.model import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(24.0)

def encodeAudio(audioFile):
    wav, sr = torchaudio.load(audioFile)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
    return codes

def decodeAudio(codes):
    encodedFrames = [(codes, None)]
    return model.decode(encodedFrames)