"""GPU smoke test for smoke-00 round.

Checks CUDA availability and times a 2048x2048 matmul on GPU.
Writes RESULTS.json to results/smoke-00/.
"""
import json
import os
import time

import torch


def main():
    out_dir = os.path.join("results", "smoke-00")
    os.makedirs(out_dir, exist_ok=True)

    gpu_available = torch.cuda.is_available()
    print(f"torch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {gpu_available}")

    device_name = ""
    matmul_ms = None
    if gpu_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"torch.cuda.get_device_name(0): {device_name}")

        # tiny GPU matmul
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")

        # warmup + sync so timing is real
        _ = a @ b
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        matmul_ms = (t1 - t0) * 1000.0
        print(f"matmul result shape: {tuple(c.shape)}, sum={c.sum().item():.2f}")
        print(f"matmul_ms: {matmul_ms:.4f}")

    results = {
        "status": "SUCCESS" if gpu_available else "FAILED",
        "scale": "probe",
        "metrics": {
            "gpu_available": bool(gpu_available),
            "device_name": device_name,
            "matmul_ms": matmul_ms,
        },
        "subject_executed": "gpu smoke test",
        "notes": (
            "2048x2048 fp32 matmul on GPU, timed after warmup + cuda.synchronize()."
            if gpu_available
            else "CUDA not available on this machine."
        ),
    }

    with open(os.path.join(out_dir, "RESULTS.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", os.path.join(out_dir, "RESULTS.json"))


if __name__ == "__main__":
    main()
