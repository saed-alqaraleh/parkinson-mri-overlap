# Computational environment and execution

**Original Keras fits:** Python 3.13.15; TensorFlow 2.20.0; Keras 3.13.2; CUDA build 12.5.1; cuDNN 9; NVIDIA A100-SXM4-80GB in the recorded environment. Frozen training: 224×224 input, batch size 64, 8 head epochs, 20 fine-tuning epochs, patience 4, learning rates 1e-4/1e-5. Keras source is provided as a portable runner whose paths and fold/seed selection are configured through environment variables; original core computational function bodies are retained.

**Original TorchVision fits:** Python 3.13; PyTorch 2.11.0+cu128; TorchVision 0.26.0+cu128; CUDA 12.8. The original `src` modules and endpoint YAML are provided. For framework CUDA wheels, use the appropriate official PyTorch index and exactly matching build tags, with no silent dependency upgrades.

**Saved-output analysis:** a separate Python 3.12 environment was recorded with NumPy 2.3.5, pandas 2.2.3 and scikit-learn 1.8.0. It is separate from model training.

**Repeat Keras execution:** for the second-seed folds 2–4, batched augmentation was used in the recorded execution; this affects execution implementation and must be preserved if comparing bitwise outputs. The batched-augmentation implementation for second-seed Keras folds 2–4 is available in `pipelines/keras/runner_batched.py`; its `make_ds` function has the same parsed syntax as the recorded execution. Dataset-discovery and runtime cache paths are configured for portability. The archived saved outputs remain authoritative.

**Performance:** end-to-end training requires the source MRI collection, released frozen manifests, compatible pretrained weights and GPU allocation. No GPU or raw-image reruns are needed for recomputing the released patient-level analyses. Estimated model-running time is not advertised because the captured conditions vary across runs.
