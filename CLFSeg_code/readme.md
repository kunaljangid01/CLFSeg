# 🧠 CLFSeg: Fuzzy-Logic-Based Medical Image Segmentation

[![Conference](https://img.shields.io/badge/Presented%20at-BMVC%202025-blue)](https://bmvc2025.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Available-blueviolet)](./Paper__CLFSeg___Medical_Image_Segmentation___BMVC_2025.pdf)

---

### 🔍 Overview

**CLFSeg** is a fuzzy-logic–driven encoder–decoder framework for **medical image segmentation**.  
It introduces a **Fuzzy-Convolution (FC)** module that combines convolutional layers and fuzzy logic to improve **boundary clarity**, **reduce uncertainty**, and **enhance robustness** while lowering computational complexity.

> 📄 _Reference:_  
> **Anshul Kaushal\*, Kunal Jangid\*, Vinod K. Kurmi**,  
> “**CLFSeg: A Fuzzy-Logic based Solution for Boundary Clarity and Uncertainty Reduction in Medical Image Segmentation**,”  
> _British Machine Vision Conference (BMVC), 2025._

---

## ⚙️ Key Features

- 🧩 **Fuzzy-Convolution Module**: Learns fuzzy membership functions to handle uncertainty and boundary ambiguity.
- ⚡ **Efficient Encoder–Decoder**: Based on convolutional and residual blocks with skip connections.
- ⚖️ **Hybrid Loss**: Combines Binary Cross-Entropy and Dice Loss for balanced optimization.
- 🔁 **Albumentations Augmentation**: Robust data augmentation pipeline.
- 🧠 **Supports both Binary & Multi-Class Segmentation**.

---

## 🧪 Architecture Overview

| Component                    | Description                                                     |
| ---------------------------- | --------------------------------------------------------------- |
| **Encoder–Decoder Backbone** | Captures multi-scale spatial features with skip connections.    |
| **Fuzzy Module**             | Uses Gaussian membership functions to model boundary fuzziness. |
| **ConvGLU**                  | Gating-based layer refining spatial feature selection.          |
| **1-ResNet Block**           | Lightweight residual pathway for fine detail recovery.          |
| **Loss Function**            | TotalLoss = BCE + DiceLoss                                      |
| **Metrics**                  | DSC, IoU, Precision, Recall, Accuracy, HD95                     |

---

## 💾 Pretrained Weights

You can download pretrained CLFSeg model weights for all benchmark datasets from the links below.  
Each model is trained using the **hybrid BCE + Dice loss** and achieves the performance reported in the BMVC 2025 paper.

| Dataset                | Filters | Format | Link         |
| ---------------------- | ------- | ------ | ------------ |
| **CVC-ColonDB**        | 17      | `.h5`  | [Download]() |
| **CVC-ClinicDB**       | 24      | `.h5`  | [Download]() |
| **ETIS-LaribPolypDB**  | 34      | `.h5`  | [Download]() |
| **ACDC (Cardiac MRI)** | 17      | `.h5`  | [Download]() |

---

## 🧬 Datasets and Results

| Dataset               | Type        | Classes | DSC (↑)    | IoU (↑) | Accuracy (↑) |
| --------------------- | ----------- | ------- | ---------- | ------- | ------------ |
| **CVC-ColonDB**       | Colonoscopy | 1       | **0.9593** | 0.9218  | 0.9945       |
| **CVC-ClinicDB**      | Colonoscopy | 1       | **0.9533** | 0.9108  | 0.9918       |
| **ETIS-LaribPolypDB** | Colonoscopy | 1       | **0.9487** | 0.9024  | 0.9946       |
| **ACDC**              | Cardiac MRI | 3       | **0.9522** | 0.9087  | —            |

> Full comparison and ablation studies are reported in **Tables 1–6** of the BMVC 2025 paper.

---

## 🖼️ Visualizations

CLFSeg offers **high interpretability** through both **activation heatmaps** and **segmentation overlays**.

### 🔹 Grad-CAM++ Interpretability

The figure below shows CLFSeg vs. DuckNet Grad-CAM++ maps on the **CVC-ColonDB** dataset.  
CLFSeg focuses more precisely on clinically relevant regions, improving sensitivity to boundaries and small structures.

<p align="center">
  <img src="https://visdomlab.github.io/CLFSeg/assets/gradcam_comparison.png" width="80%">
</p>

### 🔹 Segmentation Maps

Comparative segmentation maps demonstrate CLFSeg’s superior mask clarity and uncertainty handling on both **CVC-ColonDB** (left) and **ACDC** (right) datasets.

<p align="center">
  <img src="https://visdomlab.github.io/CLFSeg/assets/segmentation_results.png" width="80%">
</p>

---

## 🛠️ Installation

```bash
git clone https://github.com/visdomlab/CLFSeg.git
cd CLFSeg
pip install -r requirements.txt
```

---

## 🚀 Usage

🔹 Training

Uncomment the trainer section in main:

```bash
trainer = Trainer(
    IMG_HEIGHT=352,
    IMG_WIDTH=352,
    IN_CHANNELS=3,
    OUT_CHANNELS=1,
    FILTERS=17,
    EPOCHS=600,
    SAVEPATH="chkpt/",
    SAVENAME="cvc-colondb/17-Filters CLFSeg"
)
trainer.train(X_train, X_val, y_train, y_val)
```

🔹 Testing

```bash
tester = Tester(
    MODELPATH="chkpt/cvc-colondb/17-Filters CLFSeg",
    dname="cvc-colondb",
    n_classes=1
)
print(tester.test(X_test, y_test))
```

---

🧾 Citation

If you use this repository, please cite:

```bash
@inproceedings{kaushal2025clfseg,
  title={CLFSeg: A Fuzzy-Logic based Solution for Boundary Clarity and Uncertainty Reduction in Medical Image Segmentation},
  author={Kaushal, Anshul and Jangid, Kunal and Kurmi, Vinod K.},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2025}
}
```

---
