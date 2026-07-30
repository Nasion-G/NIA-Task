import { useState, useRef } from "react";
import * as ort from "onnxruntime-web";
import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile } from "@ffmpeg/util";
import "./App.css";

const SR = 24000;
const STRIDE_PRODUCT = 320;

function App() {
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const ffmpegRef = useRef(new FFmpeg());

  async function getFfmpeg() {
    const ff = ffmpegRef.current;
    if (!ff.loaded) {
      await ff.load({
        coreURL: "https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.js",
        wasmURL: "https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.wasm",
      });
    }
    return ff;
  }

  async function onUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setBusy(true);

    try {
      const ff = await getFfmpeg();
      await ff.writeFile("input", await fetchFile(file));
      await ff.exec(["-i", "input", "-ar", String(SR), "-ac", "1", "-f", "wav", "out.wav"]);
      const wavData = await ff.readFile("out.wav");

      const ctx = new AudioContext({ sampleRate: SR });
      const audio = await ctx.decodeAudioData(wavData.buffer);
      const samples = audio.getChannelData(0);

      const rawLen = samples.length;
      const remainder = rawLen % STRIDE_PRODUCT;
      const paddedLen = remainder === 0 ? rawLen : rawLen + (STRIDE_PRODUCT - remainder);

      const input = new Float32Array(paddedLen);
      input.set(samples);

      const session = await ort.InferenceSession.create("/encoder.onnx");
      const tensor = new ort.Tensor("float32", input, [1, 1, paddedLen]);
      const { codes } = await session.run({ wav: tensor });

      console.log("input seconds:", rawLen / SR, "-> codes.dims:", codes.dims);

      const [b, nq, t] = codes.dims;
      const flat = Array.from(codes.data, Number);
      const matrix = [];
      for (let i = 0; i < b; i++) {
        const batch = [];
        for (let j = 0; j < nq; j++) {
          batch.push(flat.slice((i * nq + j) * t, (i * nq + j) * t + t));
        }
        matrix.push(batch);
      }

      setStatus("sending...");
      const res = await fetch("http://localhost:8000/uploadaudio/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ codes: matrix, filename: file.name }),
      });

      const data = await res.json();
      setStatus(res.ok ? `saved as ${data.filename}` : "backend error");
    } catch (err) {
      setStatus(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <div className="upload-card">
          <h1>Audio Encoder/Decoder</h1>
          <p className="subtitle">EnCodec</p>
          <label className="upload-button">
            {busy ? "working..." : "choose file"}
            <input type="file" accept="audio/*" onChange={onUpload} disabled={busy} hidden />
          </label>
          {status && <p className="status">{status}</p>}
        </div>
      </header>
    </div>
  );
}

export default App;