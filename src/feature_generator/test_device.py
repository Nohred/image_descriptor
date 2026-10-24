"""Diagnose NVIDIA/CUDA availability in the active image-descriptors environment."""

import platform
import sys


def main() -> int:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")

    try:
        import torch
    except ImportError as error:
        print(f"ERROR: PyTorch is not installed: {error}")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA compiled in PyTorch: {torch.backends.cuda.is_built()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if not torch.backends.cuda.is_built():
        print(
            "RESULT: PyTorch is a CPU-only build. Install a CUDA-enabled "
            "PyTorch package; a driver alone is not sufficient."
        )
        return 2

    if not torch.cuda.is_available():
        print(
            "RESULT: PyTorch has CUDA support, but CUDA is unavailable. "
            "Check the NVIDIA driver, CUDA libraries, and environment."
        )
        return 3

    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
        print(f"GPU {index} capability: {torch.cuda.get_device_capability(index)}")

        values = torch.tensor([1.0, 2.0, 3.0], device=device)
        print(f"GPU {index} test tensor device: {values.device}")
        print(f"GPU {index} test tensor sum: {values.sum().item()}")

    print("RESULT: CUDA is available and PyTorch can execute on the GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
