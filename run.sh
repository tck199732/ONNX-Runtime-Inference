#!/bin/bash

set -e

USER=$(whoami)
IMAGE="$WORK_EIC/images/onnx_cuda_root_jana2.sif"

# NOTE : see sbatch.sb for GPU usage.
CMD="./build/src/inference --use_cpu"

if [ ! -f "$IMAGE" ]; then
    echo "Image not found at $IMAGE. Abort."
    exit 1
fi


WORKDIR=$(pwd)
echo $WORKDIR


echo "Using Singularity image at $IMAGE"
singularity exec --nv \
    --bind $(pwd) \
    --bind /lustre24 \
    $IMAGE \
    bash -c "
    ${CMD}
    "   





