This is the task as required.

The backend I built in Python with FastAPI. In it there is the decoding logic and also the scripts for making the .onnx encoder from EnCodec and also a test script to make sure the .onnx encoder and PyTorch EnCodec decoding works.
To run : ```bash cd backend && uv sync && uv run fastapi dev src/main.py```

The frontend I built in React. In it I used ffmpeg.wasm for the conversion to 24k MHz before it goes to ONNX with our encoder, which encodes it and sends it to the backend as a matrix.
To run: ```bash cd frontend && npm start```
