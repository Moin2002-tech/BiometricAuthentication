#!/usr/bin/env python3
"""
Convert ONNX model to TorchScript (.pt) for C++ inference with LibTorch.
Uses onnx2torch to convert ONNX -> PyTorch, then torch.jit.trace to produce TorchScript.
"""

import torch
import onnx
from onnx2torch import convert

# Paths
ONNX_PATH = "/home/moinshaikh/CLionProjects/BiometricAuthentication/models/model.onnx"
OUTPUT_PATH = "/home/moinshaikh/CLionProjects/BiometricAuthentication/models/model.torchscript.pt"

# Always trace on CPU for maximum compatibility.
# The saved TorchScript model can be moved to CUDA at load time via model.to(torch::kCUDA).
device = torch.device("cpu")
print(f"Tracing on CPU (model can be moved to CUDA at runtime)")

# 1. Load ONNX model info
onnx_model = onnx.load(ONNX_PATH)
print("ONNX model loaded successfully.")

# 2. Convert ONNX -> PyTorch (stays on CPU)
print("Converting ONNX to PyTorch using onnx2torch...")
torch_model = convert(ONNX_PATH)
torch_model.eval()
print("Conversion to PyTorch complete.")

# 3. Trace with example input (batch_size=1, 3, 112, 112) on CPU
dummy_input = torch.randn(1, 3, 112, 112)
print(f"Dummy input shape: {dummy_input.shape}")

with torch.no_grad():
    traced_model = torch.jit.trace(torch_model, dummy_input)

# 4. Save TorchScript model
traced_model.save(OUTPUT_PATH)
print(f"TorchScript model saved to: {OUTPUT_PATH}")

# 5. Verify by loading on CPU and running inference
print("Verifying TorchScript model on CPU...")
loaded = torch.jit.load(OUTPUT_PATH)
with torch.no_grad():
    output = loaded(dummy_input)
print(f"Verification output shape: {output.shape}")

# 6. Additional verification on CUDA if available
if torch.cuda.is_available():
    print("Verifying TorchScript model on CUDA...")
    loaded_cuda = torch.jit.load(OUTPUT_PATH, map_location=torch.device("cuda"))
    loaded_cuda.to(torch.device("cuda"))
    dummy_cuda = torch.randn(1, 3, 112, 112, device=torch.device("cuda"))
    with torch.no_grad():
        output_cuda = loaded_cuda(dummy_cuda)
    print(f"CUDA verification output shape: {output_cuda.shape}")

print("Conversion and verification completed successfully.")
