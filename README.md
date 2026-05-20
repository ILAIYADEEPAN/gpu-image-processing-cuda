# GPU Accelerated Image Processing using CUDA and OpenCV

## Overview

This project demonstrates GPU accelerated image processing using CUDA and OpenCV. The application performs multiple image processing operations such as grayscale conversion, Gaussian blur, and edge detection using NVIDIA GPU acceleration.

The objective of this project is to compare CPU and GPU image processing performance and understand the advantages of CUDA-based parallel computing.

---

## Technologies Used

- CUDA
- C++
- OpenCV
- NVIDIA GPU

---

## Features

- GPU accelerated grayscale conversion
- Gaussian blur using CUDA
- Edge detection using CUDA
- GPU memory transfer using GpuMat
- Performance benchmarking

---

## Project Structure

```text
gpu-image-processing-cuda/
│
├── main.cpp
├── README.md
├── sample_image.jpg
├── grayscale_output.jpg
├── blur_output.jpg
├── edge_output.jpg
├── screenshots/
│   └── terminal_output.png
└── results/
    └── execution_log.txt
```

---

## Requirements

Before running the project, install:

- CUDA Toolkit
- OpenCV with CUDA support
- CMake
- NVIDIA GPU Drivers

---

## Build Instructions

### Step 1

Clone the repository:

```bash
git clone https://github.com/yourusername/gpu-image-processing-cuda.git
```

### Step 2

Navigate into the project folder:

```bash
cd gpu-image-processing-cuda
```

### Step 3

Create build directory:

```bash
mkdir build
cd build
```

### Step 4

Build the project:

```bash
cmake ..
make
```

---

## Run Instructions

Run the application using:

```bash
./image_processor sample_image.jpg
```

---

## Output

The program generates:

- grayscale_output.jpg
- blur_output.jpg
- edge_output.jpg

The terminal also displays GPU execution timings.

Example:

```text
GPU Processing Completed!

Grayscale Time: 4.21 ms
Gaussian Blur Time: 6.18 ms
Edge Detection Time: 3.92 ms
```

---

## Screenshots

Include screenshots of:
- Original image
- Processed output images
- Terminal execution results

---

## Learning Outcomes

This project helped me understand:

- CUDA kernel execution
- GPU memory management
- Parallel image processing
- OpenCV CUDA modules
- GPU performance optimization

---

## Future Improvements

- Real-time video processing
- Deep learning integration
- Multi-GPU support
- Additional image enhancement filters

---

## Author

Ilaiya Deepan K
