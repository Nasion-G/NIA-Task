import torch
from encodec.model import EncodecModel


class Wrapped(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wav):
        emb = self.model.encoder(wav)
        codes = self.model.quantizer.encode(emb, self.model.frame_rate, self.model.bandwidth)
        return codes.transpose(0, 1)  # [n_q, B, T] -> [B, n_q, T]

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.eval()

wrapped = Wrapped(model)

dummy_input = torch.randn(1, 1, 24000 * 30)

torch.onnx.export(
    wrapped,
    dummy_input,
    "encoder.onnx",
    input_names=["wav"],
    output_names=["codes"],
    opset_version=17,
    dynamo=False,
)