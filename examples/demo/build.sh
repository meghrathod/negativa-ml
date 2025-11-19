#!/bin/sh
mkdir -p build

# A30 is compute capability 8.0
nvcc \
  --gpu-architecture=compute_80 \
  --gpu-code=sm_80 \
  --compiler-options '-fPIC' \
  -o ./build/libdemo.so \
  --shared ./demo.cu

# Link the main executable
g++ -g -O0 main.cpp -L./build -ldemo -o ./build/main \
  -Wl,-rpath,'$ORIGIN'
