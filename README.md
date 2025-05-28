# Efficient Binary Video Classification on a Local GPU-Enabled PC Using PyTorch

This repository provides an efficient binary video classification pipeline using PyTorch, optimized for local GPU-enabled PCs. It includes preprocessing and model inference tools for classifying videos into categories such as Violence and Non-Violence with high accuracy.

### Demostration Video

https://github.com/user-attachments/assets/44716d09-0db0-4be8-98f2-2625490bb68e

Real Time Detection Demo - https://youtu.be/-qG0srvAbSg?si=eSTWi9LyRru9_gJo


## **1. What is Video Classification?**
Video classification is a computer vision task that involves assigning a label or category to a video based on its content. Unlike image classification, which analyzes static frames, video classification considers temporal information across multiple frames to understand motion and context.

### **Key Applications**
- Violence detection in surveillance videos  
- Activity recognition (e.g., sports, gestures)  
- Medical diagnostics (e.g., detecting anomalies in ultrasound videos)  
- Autonomous driving (e.g., identifying road conditions)  

In this project, we focus on **binary video classification**—distinguishing between *Violence* and *Non-Violence* in videos.

---

## **2. How Video Classification Works**
Our approach uses **PyTorchVideo**, a deep learning library optimized for video understanding, and **EfficientX3D**, a lightweight 3D CNN model for efficient training on local GPUs.

### **Key Steps**
1. **Data Loading & Preprocessing**  
   - Videos are loaded and split into training/validation sets.  
   - Frames are uniformly sampled, normalized, and augmented (random cropping, flipping).  

2. **Model Training**  
   - A pre-trained `EfficientX3D` model is fine-tuned on the dataset.  
   - Binary cross-entropy loss and Adam optimizer are used.  

3. **Inference**  
   - The trained model predicts whether a new video contains violence.  

---

## **3. EfficientX3D Model Architecture**

### Overview
A lightweight 3D CNN for video/volumetric data classification with **3.8M trainable parameters** (~15.2MB memory footprint).

### Key Features
- ✅ Optimized for binary classification tasks  
- ✅ Depthwise separable 3D convolutions  
- ✅ Squeeze-Excitation attention blocks  
- ✅ Swish + ReLU activations  
- ✅ Residual connections throughout  

### Architecture Specifications
| Component          | Details                          |
|--------------------|----------------------------------|
| **Input**          | 3-channel video/volumetric data  |
| **Base Channels**  | 24 → 48 → 96 → 192 → 432         |
| **Stages**         | 5 (s1-s5) with progressive downsampling |
| **Bottleneck**     | Expansion (2.25x) → DW Conv → SE → Projection |
| **Head**           | Global pooling → FC(2048) → Dropout(0.5) → FC(400) |
| **Output**         | Linear(400→1) + BCEWithLogitsLoss |

### Parameter Breakdown
```text
Total params: 3.8M (all trainable)
Layers:
- Stem: 2 conv blocks
- Body: 4 stages (3/5/11/7 bottleneck blocks)
- Head: 2 FC layers
```
---

## **4. How to Train the Model on Your Local Machine**
### **Prerequisites**
- **Hardware**: NVIDIA GPU (e.g., RTX 2050)  
- **Software**:  
  ```bash
  pip install -r requirements.txt
  ```

### **Step-by-Step Guide**
1. **Clone this repo**
   ```git
   git clone https://github.com/BimsaraS99/Efficient-Binary-Video-Classification-on-a-Local-GPU-Enabled-PC-Using-PyTorch.git
   ```
   
2. **Prepare the Dataset**
   
   Download the dataset from this link - https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
   
   - Structure:  
     ```
     dataset/
       ├── Violence/*.mp4
       └── NonViolence/*.mp4
     ```
   - Load videos using `LabeledVideoDataset`:
     ```python
     non = glob('dataset/NonViolence/*')
     vio = glob('dataset/Violence/*')
     df = pd.DataFrame(zip(non + vio, [0]*len(non) + [1]*len(vio)), columns=['video', 'label'])
     ```

3. **Execute All Cells in training.ipynb Notebook**
   
   To ensure the training process runs smoothly, open the training.ipynb notebook and execute all cells sequentially (or use "Run All") in your preferred environment (e.g., Jupyter Notebook, Google Colab, or VS Code). This will initialize the model, load the dataset, and begin training as per the defined pipeline.

---

## **5. How to Contribute: Submitting a Pull Request**
We welcome contributions! Here’s how to get started:

### **Steps**
1. **Fork the Repository**  
   Clone and set up the project locally.

2. **Propose Improvements**  
   - Optimize data loading (e.g., multi-worker support).  
   - Experiment with larger models (X3D-M/L).  
   - Add new datasets (e.g., RWF-2000).  

3. **Submit a PR**  
   - Ensure code passes tests (`pytest`).  
   - Document changes in `README.md`.  

---
