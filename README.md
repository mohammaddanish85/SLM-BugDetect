# SLM-BugDetect

Companion code for the paper *"SLM-BugDetect: Fine-Tuning TinyLlama for Python
Bug Detection on Consumer CPUs."* A LoRA-adapted TinyLlama-1.1B that detects and
repairs common Python faults (ZeroDivisionError, IndexError, syntax violations)
while training and running entirely on a consumer CPU (Intel i5, 16 GB RAM).

## Quickstart
    pip install -r requirements.txt          # see CPU-wheel note inside
    python data/generate_dataset.py          # writes dataset.jsonl + heldout.jsonl
    python train.py                          # ~25 min on CPU; saves adapters/ + figures/loss_curve.png
    python evaluate.py --mode greedy         # reproduces Table 1 behaviour
    python evaluate.py --mode topp           # mitigation from Sec. 5.2
    python predict.py --code "for i in range(10) print(i)"

## Hyperparameters (Appendix A)
LoRA r=8, alpha=16, dropout=0.05 on q_proj/v_proj; AdamW lr=2e-4, wd=0.01;
batch 2; 3 epochs; max length 128; FP32 on CPU; labels masked -100 at pads.

## Known environment quirks (Appendix B)
- huggingface_hub warns about symlinks on Windows; the no-symlink fallback is fine.
- HF Trainer probes CUDA at init on CPU hosts; hence the manual loop in train.py.
- Older hosted GPUs (e.g. Tesla P100, CC 6.0) lack kernels in current wheels.

## Citation
If you use this code, please cite the paper (add formal citation on acceptance).

## License
MIT