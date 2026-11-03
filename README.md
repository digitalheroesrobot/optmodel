# Computer Vision Model Optimization
## A Practical Python Course: Beginner to AI Deployment Engineer

**Course Level:** Beginner to Advanced  
**Programming Language:** Python  
**Primary Focus:** Computer Vision, Model Optimization, High-Speed Inference, CPU Deployment  
**Hardware:** GPU and GPU-free computers  
**Frameworks:** PyTorch, ONNX, ONNX Runtime, OpenVINO, OpenCV

---

# PART 1. COURSE INTRODUCTION

## 1.1 What is this course?

This course teaches you how to optimize deep learning models used in computer vision applications.

We will begin with basic AI concepts and gradually learn how professional AI engineers make computer vision models:

- Faster.
- Smaller.
- More memory-efficient.
- More accurate at a given computational budget.
- Capable of running without NVIDIA GPUs.

We will use Python to implement every concept.

Each lesson contains:

1. Theory and explanations.
2. One complete Python program.
3. A real-world computer vision application.
4. Visualizations using Matplotlib or OpenCV.
5. Performance measurements.
6. Practical experiments.
7. Engineering assignments.

## 1.2 Our ultimate objective

Our final objective is to develop a real-time computer vision system that runs on an ordinary computer without an NVIDIA GPU.

For example:

A factory camera detects defective products on a conveyor belt.

The original AI model might be too slow to process images in real time.

We will optimize the model until it can process images efficiently using CPU inference.

## 1.3 Real-world applications

| Application | Computer Vision Task |
|---|---|
| Manufacturing | Industrial defect detection |
| Robotics | Object detection and localization |
| Autonomous vehicles | Object recognition |
| Healthcare | Medical image classification |
| Security | Human and vehicle detection |
| Smart cameras | Real-time object recognition |
| Retail | Product recognition |
| Agriculture | Plant disease detection |
| Logistics | Package identification |

---

# PART 2. UNDERSTANDING COMPUTER VISION

## 2.1 What is Artificial Intelligence?

Artificial Intelligence (AI) is the field of developing computer systems capable of performing tasks that normally require human intelligence.

Examples:

- Recognizing objects.
- Understanding speech.
- Translating languages.
- Predicting future events.
- Controlling robots.

## 2.2 What is Machine Learning?

Machine Learning is a branch of AI.

Instead of manually programming every rule, we teach a computer to learn patterns from data.

For example:

We provide thousands of images of cats and dogs.

The computer learns visual patterns that distinguish cats from dogs.

## 2.3 What is Deep Learning?

Deep Learning is a branch of machine learning based on neural networks containing multiple layers.

Computer vision models commonly use Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs).

## 2.4 What is Computer Vision?

Computer Vision allows computers to understand visual information.

Common tasks:

| Task | Description |
|---|---|
| Classification | Identify what an image contains |
| Object Detection | Identify objects and their locations |
| Segmentation | Identify individual object pixels |
| Pose Estimation | Estimate human or robot body positions |
| OCR | Recognize text in images |
| Tracking | Track objects across video frames |

## 2.5 Training versus Inference

These are two important AI concepts.

### Training

Training teaches a neural network using a dataset.

```text
Training Images
      |
      v
Neural Network
      |
      v
Learning Process
      |
      v
Trained Model
```

### Inference

Inference uses an already-trained model to make predictions.

```text
New Image
    |
    v
Preprocessing
    |
    v
Trained Model
    |
    v
Prediction
```

This course focuses primarily on inference optimization.

---

# PART 3. UNDERSTANDING MODEL OPTIMIZATION

## 3.1 Why optimize AI models?

Consider a computer vision model with the following performance:

| Metric | Original Model |
|---|---:|
| Inference latency | 150 ms |
| FPS equivalent | 6.67 |
| Model size | 200 MB |
| Memory consumption | 1.5 GB |
| Hardware | CPU |

This model might be unsuitable for real-time applications.

Our objective is to improve its performance.

## 3.2 Important performance metrics

### Latency

Latency is the time required to process one image.

For example:

```text
Image Input
    |
    v
AI Processing
    |
    v
Prediction

Total time: 50 ms
```

### FPS

FPS means Frames Per Second.

Formula:

\[
FPS = \frac{1000}{Latency(ms)}
\]

Examples:

| Latency | FPS Equivalent |
|---:|---:|
| 10 ms | 100 |
| 20 ms | 50 |
| 50 ms | 20 |
| 100 ms | 10 |
| 200 ms | 5 |

This formula describes sequential single-image inference. Actual video throughput can differ because of preprocessing, postprocessing, batching, and pipelining.

### Model Size

The amount of storage required by the trained model.

### Memory Consumption

The RAM required to load and execute the model.

### Accuracy

The model's ability to make correct predictions.

An optimized model must maintain acceptable accuracy.

## 3.3 Main optimization techniques

| Technique | Purpose |
|---|---|
| Lightweight Architecture | Use fewer computations |
| Input Resolution Reduction | Reduce image processing cost |
| ONNX Conversion | Use portable inference runtimes |
| Quantization | Reduce numerical precision |
| Model Pruning | Remove unnecessary parameters |
| Knowledge Distillation | Train a smaller student model |
| Operator Fusion | Combine operations |
| Hardware Optimization | Use efficient CPU instructions |
| Multithreading | Improve execution throughput |

---

# PART 4. DEVELOPMENT ENVIRONMENT

## 4.1 Recommended project structure

```text
computer_vision_optimization/
│
├── README.md
│
├── requirements.txt
│
├── datasets/
│   ├── images/
│   └── videos/
│
├── models/
│
├── outputs/
│
├── visualizations/
│
└── lessons/
    │
    ├── 01_baseline_inference.py
    ├── 02_measure_inference_speed.py
    ├── 03_model_architecture.py
    ├── 04_input_resolution.py
    ├── 05_model_comparison.py
    ├── 06_onnx_conversion.py
    ├── 07_onnx_runtime.py
    ├── 08_quantization_fp16_int8.py
    ├── 09_dynamic_quantization.py
    ├── 10_openvino_inference.py
    ├── 11_model_pruning.py
    ├── 12_knowledge_distillation.py
    ├── 13_yolo_optimization.py
    ├── 14_parallel_inference.py
    └── 15_final_edge_ai_project.py
```

## 4.2 Install Python

Recommended Python version: 3.10–3.12.

Create the project:

```bash
mkdir computer_vision_optimization

cd computer_vision_optimization

python3 -m venv .venv

source .venv/bin/activate
```

## 4.3 Install libraries

Install CPU-only PyTorch:

```bash
pip install torch torchvision \
--index-url https://download.pytorch.org/whl/cpu
```

Install other libraries:

```bash
pip install numpy matplotlib pillow psutil

pip install opencv-python

pip install onnx onnxruntime

pip install openvino
```

For later lessons:

```bash
pip install ultralytics

pip install scikit-learn
```

Verify installation:

```python
import torch
import torchvision
import cv2
import onnxruntime
import openvino

print("PyTorch:", torch.__version__)

print("Torchvision:", torchvision.__version__)

print("OpenCV:", cv2.__version__)

print("ONNX Runtime:", onnxruntime.__version__)

print("OpenVINO:", openvino.__version__)

print("CUDA available:", torch.cuda.is_available())
```

---

# PART 5. COMPLETE PRACTICAL COURSE

# LESSON 01 — Baseline Computer Vision Inference

**Python File:** `01_baseline_inference.py`

## Learning Objectives

Understand how a pretrained computer vision model works.

Learn:

- What is a pretrained model?
- How to load a neural network.
- How to load an image.
- How to preprocess an image.
- How to perform inference.
- How to interpret predictions.
- How to measure inference latency.

## Real-World Application

A smart camera recognizes objects.

Example:

```text
Camera Image
     |
     v
MobileNetV3
     |
     v
Prediction
     |
     v
Golden Retriever
```

## Model

MobileNetV3-Small.

## Visualization

Display:

1. Original image.
2. Top five classification predictions.
3. Prediction confidence.
4. Inference latency histogram.

## Experiment

Run inference 100 times.

Record:

- Average latency.
- Median latency.
- P95 latency.
- FPS.

## Engineering Challenge

Run the same model using different CPU thread counts.

Compare the results.

---

# LESSON 02 — Measuring AI Inference Performance

**Python File:** `02_measure_inference_speed.py`

## Learning Objectives

Learn how professional AI engineers measure inference performance.

Understand:

- Latency.
- FPS.
- Warmup.
- Benchmarking.
- CPU utilization.
- RAM consumption.
- P50, P95 and P99 latency.

## Real-World Application

A real-time CCTV system must process camera images continuously.

We need to determine whether our model is fast enough.

## Python Experiment

Build a benchmarking application that:

1. Loads a pretrained model.
2. Loads a collection of images.
3. Runs inference repeatedly.
4. Measures each inference.
5. Measures CPU utilization.
6. Measures RAM consumption.
7. Generates performance charts.

## Visualization

Generate:

- Latency distribution histogram.
- FPS comparison chart.
- CPU utilization chart.
- Memory usage chart.

## Engineering Challenge

Compare performance with:

- 1 CPU thread.
- 2 CPU threads.
- 4 CPU threads.
- Default CPU thread configuration.

Determine which configuration performs best.

---

# LESSON 03 — Understanding Neural Network Architecture

**Python File:** `03_model_architecture.py`

## Learning Objectives

Understand how computer vision models are constructed.

Study:

- Neural network layers.
- Convolution.
- Activation functions.
- Pooling.
- Feature extraction.
- Fully connected layers.
- Parameters.
- FLOPs.

## Real-World Application

An industrial camera identifies different products.

Understand how a neural network extracts visual features.

## Python Experiment

Load:

- MobileNetV3-Small.
- ResNet18.
- EfficientNet-B0.

Inspect:

- Number of parameters.
- Model structure.
- Layer output shapes.
- Approximate computational complexity.

## Visualization

Create:

1. Model architecture visualization.
2. Parameter count comparison.
3. FLOPs comparison.
4. Model size comparison.

## Engineering Challenge

Explain why two models with similar parameter counts can have different inference speeds.

---

# LESSON 04 — Input Resolution Optimization

**Python File:** `04_input_resolution.py`

## Learning Objectives

Understand how image resolution affects model performance.

Learn:

- Image resizing.
- Tensor dimensions.
- Computational complexity.
- Resolution versus accuracy.
- Input preprocessing.

## Real-World Application

A factory camera detects defective products.

The original image is 1920 × 1080.

However, the model might only need a 320 × 320 image.

## Python Experiment

Run the same model with multiple resolutions:

```text
1920 x 1080
1024 x 1024
640 x 640
320 x 320
224 x 224
```

For classification models, use appropriate preprocessing and compatible model input sizes.

## Visualization

Generate:

- Original image.
- Resized images.
- Inference latency comparison.
- Resolution versus FPS chart.

## Engineering Challenge

Determine how much inference speed improves when reducing image resolution.

Investigate how classification accuracy changes.

---

# LESSON 05 — Comparing Lightweight Models

**Python File:** `05_model_comparison.py`

## Learning Objectives

Understand why choosing the right architecture is important.

Compare:

- MobileNetV3-Small.
- MobileNetV3-Large.
- ResNet18.
- EfficientNet-B0.
- SqueezeNet.

## Real-World Application

Deploy an image classification system on a low-cost industrial computer.

We need to select the most appropriate model.

## Python Experiment

For each model, measure:

- Parameter count.
- Model file size.
- Inference latency.
- FPS.
- Peak memory usage.
- Classification accuracy on a common test dataset.

## Visualization

Create a comparison dashboard:

```text
MODEL COMPARISON

MobileNetV3-Small  | FAST
MobileNetV3-Large  | MEDIUM
ResNet18           | MEDIUM
EfficientNet-B0    | MEDIUM
SqueezeNet         | FAST
```

Use actual measurements instead of assuming these rankings.

## Engineering Challenge

Select the best model for an ordinary CPU computer.

Explain your decision based on speed and accuracy.

---

# LESSON 06 — Converting PyTorch Models to ONNX

**Python File:** `06_onnx_conversion.py`

## Learning Objectives

Understand ONNX.

Learn:

- What is ONNX?
- Why use ONNX?
- PyTorch model export.
- Input and output tensors.
- Dynamic dimensions.
- ONNX model inspection.

## Real-World Application

A model trained using PyTorch needs to run on different computer platforms.

ONNX provides a portable model representation.

## Python Experiment

Convert MobileNetV3-Small:

```text
PyTorch Model
      |
      v
   ONNX Export
      |
      v
   model.onnx
```

Verify that the exported model produces predictions consistent with the original model.

## Visualization

Display:

- Original model output.
- ONNX model output.
- Prediction differences.

## Engineering Challenge

Export a model with a dynamic batch dimension.

Understand how dynamic input shapes work.

---

# LESSON 07 — Fast CPU Inference with ONNX Runtime

**Python File:** `07_onnx_runtime.py`

## Learning Objectives

Understand how ONNX Runtime executes models.

Learn:

- Inference sessions.
- CPU execution provider.
- Input tensors.
- Output tensors.
- Execution providers.
- Thread configuration.

## Real-World Application

A computer vision application must run on a PC without an NVIDIA GPU.

## Python Experiment

Load the ONNX model:

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "models/mobilenet.onnx",
    providers=["CPUExecutionProvider"]
)
```

Run inference and benchmark performance.

## Visualization

Compare:

- PyTorch CPU inference.
- ONNX Runtime CPU inference.

Generate:

- Latency chart.
- FPS chart.
- Memory comparison.

## Engineering Challenge

Experiment with ONNX Runtime intra-op and inter-op thread settings.

Determine the best configuration for your computer.

---

# LESSON 08 — Model Quantization: FP32, FP16 and INT8

**Python File:** `08_quantization_fp16_int8.py`

## Learning Objectives

Understand numerical precision in deep learning.

Study:

- FP32.
- FP16.
- BF16.
- INT8.
- Quantization scales.
- Zero points.
- Quantization error.

## Real-World Application

A smart camera has limited memory and processing power.

We need a smaller model that can run efficiently on CPU.

## Python Experiment

Create and compare:

```text
Original FP32 Model
        |
        v
    Quantization
        |
        +------ FP16
        |
        +------ INT8
```

Important:

FP16 and INT8 do not necessarily accelerate every CPU. Hardware and runtime support determine actual performance.

## Visualization

Generate:

- Model size comparison.
- Inference latency comparison.
- Prediction confidence comparison.
- Quantization error visualization.

## Engineering Challenge

Determine which precision gives the best speed, memory and accuracy trade-off.

---

# LESSON 09 — Dynamic Quantization and Static Quantization

**Python File:** `09_dynamic_quantization.py`

## Learning Objectives

Understand different quantization methods.

Study:

### Dynamic Quantization

Weights are quantized, while activation quantization parameters are determined during execution.

### Static Quantization

Weights and activation quantization parameters are determined using calibration data.

Study:

- Calibration datasets.
- Representative samples.
- Quantization-aware training.
- Quantization error.

## Real-World Application

A manufacturing company wants to deploy a lightweight model on Intel CPU computers.

## Python Experiment

Compare:

- FP32.
- Dynamic INT8.
- Static INT8.

Note: PyTorch dynamic quantization is particularly useful for supported linear layers; it is not a universal CNN acceleration method.

## Visualization

Display:

- Model size.
- Inference latency.
- Accuracy.
- Quantization error.

## Engineering Challenge

Explain why calibration data matters for static quantization.

---

# LESSON 10 — Intel OpenVINO Optimization

**Python File:** `10_openvino_inference.py`

## Learning Objectives

Learn how to deploy AI models using Intel OpenVINO.

Study:

- OpenVINO architecture.
- Model conversion.
- Model compilation.
- CPU execution.
- Inference requests.
- Performance hints.
- INT8 optimization.

## Real-World Application

Deploy object recognition on an ordinary Intel mini PC.

## Python Experiment

Load an ONNX model using OpenVINO.

```python
import openvino as ov

core = ov.Core()

model = core.read_model(
    "models/mobilenet.onnx"
)

compiled_model = core.compile_model(
    model,
    "CPU"
)
```

Run inference.

Compare OpenVINO with ONNX Runtime and PyTorch.

## Visualization

Generate:

- Framework comparison chart.
- Latency chart.
- FPS comparison.
- Memory usage chart.

## Engineering Challenge

Determine whether OpenVINO provides a performance advantage on your Intel CPU.

---

# LESSON 11 — Model Pruning

**Python File:** `11_model_pruning.py`

## Learning Objectives

Understand how unnecessary model weights can be removed or made sparse.

Study:

- Weight pruning.
- Structured pruning.
- Unstructured pruning.
- Sparsity.
- Fine-tuning after pruning.

## Real-World Application

A factory computer has limited memory and computational resources.

We want to reduce model complexity.

## Python Experiment

Create:

```text
Original Model
      |
      v
Pruning
      |
      v
Fine-Tuning
      |
      v
Optimized Model
```

Compare original and pruned models.

## Visualization

Generate:

- Weight distribution.
- Sparsity visualization.
- Model size comparison.
- Accuracy comparison.

## Engineering Challenge

Investigate why removing weights does not always increase inference speed.

Understand the difference between sparse weights and smaller dense architectures.

---

# LESSON 12 — Knowledge Distillation

**Python File:** `12_knowledge_distillation.py`

## Learning Objectives

Understand how a large model can teach a smaller model.

Study:

- Teacher model.
- Student model.
- Soft targets.
- Temperature.
- Distillation loss.
- Knowledge transfer.

## Real-World Application

A powerful model runs too slowly on a low-cost computer.

We want a smaller model that preserves as much of its performance as possible.

## Python Experiment

Use:

```text
Large Teacher Model
        |
        v
   Soft Predictions
        |
        v
 Small Student Model
        |
        v
  Lightweight Model
```

Train a student model using knowledge distillation.

## Visualization

Generate:

- Teacher accuracy.
- Student accuracy.
- Training loss.
- Model size comparison.
- Inference speed comparison.

## Engineering Challenge

Investigate how much accuracy the student model can preserve while reducing computational requirements.

---

# LESSON 13 — YOLO Object Detection Optimization

**Python File:** `13_yolo_optimization.py`

## Learning Objectives

Move from image classification to object detection.

Study:

- YOLO architecture.
- Bounding boxes.
- Object confidence.
- Class prediction.
- Non-Maximum Suppression.
- Input resolution.
- Detection latency.

## Real-World Application

An industrial camera identifies defective products moving along a conveyor belt.

## Python Experiment

Use a pretrained lightweight YOLO model.

Run detection on:

- Images.
- Videos.
- Webcam streams.

Compare:

- Original PyTorch model.
- ONNX model.
- ONNX Runtime CPU.
- OpenVINO CPU.

## Visualization

Display:

- Original camera image.
- Bounding boxes.
- Object labels.
- Confidence scores.
- FPS counter.

## Engineering Challenge

Optimize detection speed while maintaining acceptable detection accuracy.

---

# LESSON 14 — Parallel Inference and Multiple Cameras

**Python File:** `14_parallel_inference.py`

## Learning Objectives

Understand how to process multiple image streams.

Study:

- Python threading.
- Multiprocessing.
- Producer-consumer architecture.
- Inference queues.
- Batch inference.
- Throughput.
- Pipeline latency.

## Real-World Application

A factory has four cameras.

Each camera monitors a different production line.

The computer must process all four streams.

## Python Experiment

Build a multi-camera simulation.

Compare:

- Sequential inference.
- Multithreaded inference.
- Batched inference.

## Visualization

Display:

- Four camera feeds.
- Per-camera FPS.
- Total throughput.
- CPU utilization.
- Processing latency.

## Engineering Challenge

Determine the best strategy for processing multiple cameras using CPU-only hardware.

---

# LESSON 15 — FINAL PROJECT: GPU-FREE EDGE AI SYSTEM

**Python File:** `15_final_edge_ai_project.py`

## Project Objective

Build a complete real-time computer vision deployment system.

The system must operate without an NVIDIA GPU.

## System Architecture

```text
        USB CAMERA
             |
             v
       IMAGE CAPTURE
             |
             v
        PREPROCESSING
             |
             v
      OPTIMIZED YOLO
             |
             v
       CPU INFERENCE
             |
             v
      OBJECT DETECTION
             |
             v
      POSTPROCESSING
             |
             v
       VISUALIZATION
             |
             v
        FPS MONITOR
```

## Required Features

1. Load a pretrained YOLO model.
2. Capture live camera frames.
3. Resize and preprocess images.
4. Load an optimized ONNX model.
5. Run CPU inference.
6. Display object detection results.
7. Display confidence scores.
8. Display FPS.
9. Measure inference latency.
10. Measure memory consumption.
11. Save detection results.
12. Compare optimized and original models.

## Performance Evaluation

Create a final report:

| Metric | Original Model | Optimized Model |
|---|---:|---:|
| Model Size | Measure | Measure |
| Average Latency | Measure | Measure |
| P95 Latency | Measure | Measure |
| FPS | Measure | Measure |
| Memory Usage | Measure | Measure |
| Detection Accuracy | Measure | Measure |

## Final Engineering Challenge

Deploy your application on an ordinary computer without an NVIDIA GPU.

Demonstrate:

- Live camera detection.
- Real-time visualization.
- CPU inference.
- Performance benchmarking.
- Model optimization.
- Accuracy comparison.

---

# PART 6. ADDITIONAL ENGINEERING KNOWLEDGE

After completing the 15 lessons, study these advanced topics.

## 6.1 Computational Complexity

Understand:

- FLOPs.
- MACs.
- Parameters.
- Memory bandwidth.
- Cache utilization.
- CPU vectorization.

## 6.2 Hardware Optimization

Understand:

- Intel CPU architecture.
- SIMD instructions.
- AVX2.
- AVX-512.
- ARM NEON.
- Hardware acceleration.
- Memory bottlenecks.

## 6.3 Inference Runtime Optimization

Study:

- ONNX Runtime execution providers.
- OpenVINO execution.
- Graph optimization.
- Operator fusion.
- Thread scheduling.
- Memory allocation.

## 6.4 Edge AI Deployment

Learn to deploy models on:

- Ordinary desktop computers.
- Intel mini PCs.
- Raspberry Pi.
- ARM-based computers.
- Industrial PCs.

---

# PART 7. HOW TO STUDY THIS COURSE

Follow this learning process for every lesson.

### Step 1 — Understand the theory

Read the explanation before running the code.

### Step 2 — Run the Python program

Execute the complete Python file.

### Step 3 — Examine the visualization

Understand what the graphs and images tell you.

### Step 4 — Record performance

Save latency, FPS, memory consumption and accuracy measurements.

### Step 5 — Experiment

Change one parameter at a time.

### Step 6 — Compare results

Determine whether your changes improved performance.

### Step 7 — Explain the engineering result

Be able to explain:

- What you changed.
- Why you changed it.
- What improved.
- What became worse.
- Whether accuracy was affected.

---

# PART 8. FINAL LEARNING OUTCOMES

After completing this course, you should be able to:

- Understand computer vision models.
- Load pretrained AI models.
- Measure inference performance.
- Compare neural network architectures.
- Optimize image resolution.
- Convert PyTorch models to ONNX.
- Deploy models using ONNX Runtime.
- Apply model quantization.
- Use Intel OpenVINO.
- Understand model pruning.
- Train lightweight student models.
- Optimize YOLO object detection.
- Build parallel inference pipelines.
- Deploy GPU-free computer vision applications.

## Final Goal

Become capable of taking an existing computer vision model and engineering an optimized deployment for real-world hardware.

**Remember: A successful AI engineer doesn't just build a model that works. They build a model that works efficiently on the hardware available.**