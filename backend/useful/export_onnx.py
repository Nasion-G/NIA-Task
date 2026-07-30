import math
import torch
import torch.nn.functional as F
from encodec.model import EncodecModel
import encodec.modules.conv as conv_mod


def get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total=0):
    length = torch.tensor(x.shape[-1])
    n_frames = (length - kernel_size + padding_total).float() / stride + 1
    ideal_length = (torch.ceil(n_frames).long() - 1) * stride + (kernel_size - padding_total)
    return (ideal_length - length).item() if not torch.onnx.is_in_onnx_export() else (ideal_length - length)


conv_mod.get_extra_padding_for_conv1d = get_extra_padding_for_conv1d


class Wrapped(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wav):
        emb = self.model.encoder(wav)
        codes = self.model.quantizer.encode(emb, self.model.frame_rate, self.model.bandwidth)
        return codes.transpose(0, 1)

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.eval()
for m in model.modules():
    if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
        torch.nn.utils.remove_weight_norm(m)

wrapped = Wrapped(model)
wrapped.eval()

dummy_input = torch.randn(1, 1, 24000 * 2)

with torch.no_grad():
    torch.onnx.export(
        wrapped,
        dummy_input,
        "encoder.onnx",
        input_names=["wav"],
        output_names=["codes"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "wav": {0: "batch", 2: "time"},
            "codes": {0: "batch", 2: "frames"},
        },
    )