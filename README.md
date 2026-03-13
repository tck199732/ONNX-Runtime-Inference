# ONNX Runtime Inference

## Introduction

ONNX Runtime C++ inference example for cluster reconstruction using CPU and CUDA.

## Usages

### C++ Inference

#### Build Docker Image

```bash
$ singularity build ${IMG_DIR}/image.sif singularity/onnx_cuda_root_jana2.def
```

This takes for more than 30 mins. Go grab coffee. For convenience, also create symbolic link `ln -s ${IMG_DIR}/image.sif image.sif`.

#### Build Example

see `build.sh`

#### Run Example

- for cpu usage, see `run.sh`.
- for job submission on ifarm, `sbatch.sb`.

