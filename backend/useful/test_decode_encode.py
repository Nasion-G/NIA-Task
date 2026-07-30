import torch
import onnxruntime as ort
import torchaudio
from encodec.utils import convert_audio
from src.codec import model, decodeAudio
print("imports done", flush=True)

wav, sr = torchaudio.load("gc.wav")
print("loaded wav", flush=True)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)

target_len = 24000 * 30
if wav.shape[-1] < target_len:
    wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[-1]))
else:
    wav = wav[..., :target_len]
print("padded", flush=True)

sess = ort.InferenceSession("../encoder.onnx")
print("onnx session created", flush=True)
onnx_codes = sess.run(None, {"wav": wav.numpy()})[0]
print("onnx run done", flush=True)

codes_tensor = torch.from_numpy(onnx_codes)
decoded_wav = decodeAudio(codes_tensor)
print("decoded", flush=True)

print("decoded_wav shape:", decoded_wav.shape)
torchaudio.save("onnx_roundtrip.wav", decoded_wav.squeeze(0), model.sample_rate)
print("Saved onnx_roundtrip.wav, shape:", decoded_wav.shape)