"""Faithful merge of a QLoRA adapter.

A LoRA adapter trained with QLoRA was optimised on top of the NF4-QUANTIZED base weights. Merging it into the
original full-precision weights (the usual `merge_and_unload()` on an FP16 base) therefore changes the function the
adapter was trained for; on the identifier-free Edge-IIoTset model this cost 0.07 macro-F1 (FP16) and 0.14 (INT8).
The faithful export merges the adapter into the DE-QUANTIZED NF4 base (identical NF4 configuration as in training,
incl. double quantization), so the merged FP16 model reproduces the trained QLoRA classifier, and every deployed
format (INT8, GGUF Q8_0 / Q4_K_M) is derived from that merged model.
"""
import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

NF4 = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)


def _manual_dequantize(model):
    import bitsandbytes as bnb
    for _, mod in list(model.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, bnb.nn.Linear4bit):
                W = bnb.functional.dequantize_4bit(child.weight.data, child.weight.quant_state).to(torch.float16)
                lin = torch.nn.Linear(child.in_features, child.out_features, bias=child.bias is not None,
                                      device=W.device, dtype=torch.float16)
                lin.weight.data.copy_(W)
                if child.bias is not None: lin.bias.data.copy_(child.bias.data.to(torch.float16))
                setattr(mod, cn, lin)
    return model


def load_nf4_base(model_id, num_labels, pad_token_id):
    b = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels, device_map={"": 0},
                                                           quantization_config=BitsAndBytesConfig(**NF4))
    b.config.pad_token_id = pad_token_id
    return b


def dequantized_merge(model_id, num_labels, pad_token_id, adapter):
    """NF4 base -> de-quantize -> FP16 -> merge adapter. Returns an FP16 classifier on GPU."""
    from peft import PeftModel
    b = load_nf4_base(model_id, num_labels, pad_token_id)
    try:
        b = b.dequantize()
    except Exception as e:
        print("transformers dequantize() unavailable, manual de-quantization:", e, flush=True)
        b = _manual_dequantize(b)
    b = b.to(torch.float16)
    for k in ("is_loaded_in_4bit", "is_quantized"):
        if hasattr(b, k):
            try: setattr(b, k, False)
            except Exception: pass
    if hasattr(b, "hf_quantizer"): b.hf_quantizer = None
    if hasattr(b.config, "quantization_config"): delattr(b.config, "quantization_config")
    m = PeftModel.from_pretrained(b, str(adapter)).merge_and_unload()
    return m.to(torch.float16).eval()
