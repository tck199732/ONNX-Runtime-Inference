#!/bin/bash

# modify image pth accordingly
IMAGE="./image.sif"

if [ ! -f "$IMAGE" ]; then
    echo "Image not found at $IMAGE. Cannot build."
    exit 1
fi

echo "Using Singularity image at $IMAGE"
singularity exec \
    --bind $(pwd) \
    $IMAGE \
    bash -c "
    cmake -B build -S .
    cmake --build build -j 8
    "   
