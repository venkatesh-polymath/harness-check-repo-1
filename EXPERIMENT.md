# SMOKE TEST (not a research experiment)
Do exactly this and nothing more, in under 2 minutes:
1. Run `nvidia-smi` (Bash).
2. In python check torch.cuda.is_available() and torch.cuda.get_device_name(0).
3. Do a tiny GPU matmul: c = torch.randn(2048,2048,device='cuda') @ torch.randn(2048,2048,device='cuda'); time it.
4. Write ./results/smoke-00/RESULTS.json with:
   {"status":"SUCCESS","scale":"probe",
    "metrics":{"gpu_available":<bool>,"device_name":"<str>","matmul_ms":<float>},
    "subject_executed":"gpu smoke test","notes":"<short>"}
5. Write ./results/smoke-00/LOG.md with what you observed.
Do NOT train models or download datasets.
