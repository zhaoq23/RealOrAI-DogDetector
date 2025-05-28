# Bark or Bot?
## Detecting Real vs. AI Dog Images

<img src="https://i.ytimg.com/vi/b7rB7VFl_I4/maxresdefault.jpg" alt="Dog vs Robot" width="400"/>

# Table of Contents
1. [Problem Statement](#problem-statement)
2. [Data and Repository Structure](#data-and-repository-structure)
3. [Methodology](#methodology)<br>
	3.1. [Environment setup](#environment-setup)<br>
	3.2. [Load and preprocess image data](#load-and-preprocess-image-data)<br>
	3.3. [Add edge channel to images](#add-edge-channel-to-images)<br>
	3.4. [CNN model experiments](#cnn-model-experiments)<br>
	3.5. [Train Bayesian MLP and evaluate](#train-bayesian-mlp-and-evaluate)<br>
	3.6. [Precompute feature tensors](#precompute-feature-tensors)<br>
 	3.7. [Widget](#widget)<br>
4. [Results](#results)
5. [Final Takeaways and Next Steps](#final-takeaways-and-next-steps)

# Problem Statement

AI-generated content is getting eerily real, especially with the recent release of Google Veo3, which can now generate hyper-realistic videos.

This project aims to classify real dog images vs. AI-generated dog images using a combination of deep learning feature extraction, enhanced input processing, and Bayesian modeling to incorporate uncertainty in predictions. Specifically, our goals include:

   - Build a reliable classifier to distinguish between real and AI-generated dog images.
   - Enhance model performance through feature engineering (e.g., edge channel).
   - Use Bayesian methods for uncertainty-aware predictions.
   - Evaluate multiple models (CNN baseline, MLP variants with basic and engineered features) using consistent feature sets.

# Difference

**Lack of Sensor Noise in AI-Generated Dog Images**

Nightingale and Farid (2022) observed that AI-generated images often contain mismatches in local features. We similarly find that AI-generated dog images may show anomalies like strange or missing legs, which are rare in real photos and useful for distinguishing generated content.  
Reference: Nightingale & Farid, 2022. "Detection of GAN-generated imagery using statistical inconsistencies."

<img src="https://github.com/user-attachments/assets/acff67f6-125b-474d-b290-62717060685f" width="400"/> <br><br>

**Frequency Artifacts in AI-Generated Images**

According to Durall et al. (2020), AI-generated images tend to exhibit unnatural frequency patterns. These differ from the smooth transitions in real images and can be leveraged to detect synthetic content.
Reference: Durall et al., 2020. "Unmasking DeepFakes with Simple Features." [arXiv]

<img src="https://github.com/user-attachments/assets/593eef45-dfdb-4bb0-b803-853a7445ad5a" width="600"/> <br><br>

AI-generated dog images may look too smooth, have strange body shapes, and show light or shadows that do not make sense. Sometimes, AI dogs also make movements or poses that real dogs cannot do. Our model uses these differences to tell real and AI-generated dog images apart.

# Data and Repository Structure 
## 📁 Dataset Structure
We use the [AI-Generated Dogs vs. Real Dogs Dataset](https://www.kaggle.com/datasets/albertobircoci/ai-generated-dogs-jpg-vs-real-dogs-jpg), which contains a collection of real and AI-generated dog images. The data is organized into three main subsets: **Train**, **Validation**, and **Test**, each with:

- `Images/`: Input JPEG images  
- `Labels/`: Ground-truth labels (`0` = real dog, `1` = AI-generated)

We provide two versions of the dataset:

### 📊 Dataset Statistics

| Subset      | Version     | Total Images | AI-generated (`1`) | Real (`0`) |
|-------------|-------------|--------------|---------------------|------------|
| **Train**   | Full        | 18,605       | 4,200               | 14,405     |
|             | Cutted      | 344          | 172                 | 172        |
| **Valid**   | Full        | 5,317        | 1,200               | 4,117      |
|             | Cutted      | 100          | 50                  | 50         |
| **Test**    | Full        | 2,658        | 600                 | 2,058      |
|             | Cutted      | 50           | 25                  | 25         |

> 💡 The **Cutted** version is a lightweight, balanced subset useful for quick debugging, prototyping, or reproducibility testing, while the **Full** version supports large-scale training and evaluation.


## Repository Structure
Our GitHub repository is organized as follows:

```text
RealOrAI-DogDetector/
├── 1_data_process.ipynb           # Preprocesses image data and extracts features
├── 2_Simple_MLP.ipynb             # Trains a simple MLP classifier
├── 3_Bayesian_MLP_Model_Train.ipynb  # Trains a Bayesian MLP model with uncertainty estimation
├── 4_method_explore.ipynb         # Explores CNN architectures (e.g., ResNet50) and compares methods
├── 5_Enhanced_Bayesian_MLP.ipynb  # Adds edge detection (RGBA) for improved Bayesian performance
├── 6_classify_new_image.ipynb     # Classifies a custom image (real vs AI-generated)
├── models/
│   ├── baseline_model.pt          # Trained baseline model with standard features
│   └── plusdiff_model.pt          # Trained enhanced model with edge-aware features
└── README.md                      # Project documentation (this file)
```

# Methodology

This project followed a structured yet flexible approach to solving the problem of classifying real vs. AI-generated dog images. We explored multiple model types and feature engineering techniques to find the best balance between accuracy, speed, and interpretability.

## Environment setup

We use Hugging Face's transformers library to import an image feature extractor and model backbone for efficient transfer learning.

```
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoModel
```

## Load and preprocess image data
`data_clean.ipynb`

We started with `data_clean.ipynb`, which handled all the image preprocessing. We loaded our dataset of real and fake dog images, resized them to 224x224 pixels, normalized them using standard ImageNet stats, and wrapped everything in PyTorch Dataloaders. This gave us a clean, consistent input pipeline to use across all the different modeling approaches.

All images were transformed into normalized tensors, ready for CNNs or embedding extraction.

## Precompute feature tensors 
`features_train_*.pt`
`features_val_*.pt `
`features_test_*.pt`

We saved all our features as .pt files to speed things up during experimentation. The features_train_full.pt, features_val_full.pt, and features_test_full.pt files contained the original CNN-based embeddings from RGB images. 

The features_train_plus.pt, features_val_plus.pt, and features_test_plus.pt files held the same embeddings but with the edge channel added.

   - These precomputed tensors let us train MLPs in seconds instead of minutes
   - The “plus” versions were consistently more effective than the originals
   - We also made “cutted” versions of these files with smaller samples to use for quick debugging

Due to large file sizes, the **precomputed feature tensors** for training, validation, and testing are hosted externally.

👉 [Click here to access the feature files on Google Drive](https://drive.google.com/drive/folders/1gDfxVEp0vZ-3u6BigSQOoL-PjhERqRJq?usp=drive_link)

To use:
1. Download the `.pt` files from the Google Drive link above.
2. Place them in the root directory or update the paths in the notebooks accordingly.

## Simple MLP Model

`Simple_MLP.ipynb`

We use a simple Multi-Layer Perceptron (MLP) with two hidden layers (128 and 64 units) and dropout for regularization. The model takes 384-dimensional feature vectors as input and predicts whether an image is real or AI-generated.

- **Input**: 384-d feature vector  
- **Architecture**: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Output
- **Validation Accuracy**: 98.98%  
- **Training Accuracy**: 98.72%  

This model is efficient and performs well on extracted image features.

## Train Bayesian MLP and evaluate
`Bayesian MLP Model_Train.ipynb`

Once we had features extracted from the images, we trained our classifiers. This would served as our baseline model.

We compared two types: a standard MLP and a Bayesian version that uses dropout to estimate uncertainty. 

These models didn’t operate on raw images—they were trained on precomputed feature tensors (either from RGB images or the edge-enhanced RGBA ones).
   - The baseline MLP trained on regular RGB features achieved ~79% accuracy
   - The Bayesian MLP added a bit of robustness and interpretability ~80% accuracy
   - - The big leap came when we fed it the enhanced (RGBA) features. With the plus features, the Bayesian MLP hit ~86% accuracy and gave us confidence scores for each prediction
       
The Bayesian MLP (RGBA features) model made it easier to spot borderline cases or flag predictions the model wasn’t sure about.

## CNN Model Experiments (End-to-End Models with ResNet50)
`method_explore.ipynb`

To go beyond the DINO+MLP baseline, we explored a series of **end-to-end CNN models** based on **ResNet50**. Unlike our MLP models that rely on pre-extracted features, these CNNs **learn directly from images**, combining feature extraction and classification in one pipeline.

Because of their heavy training cost (dozens of times slower than MLPs), we used two validation strategies:
- **Cutted dataset**: 97% validation accuracy with standard ResNet50
- **Random subsamples**: 95% accuracy on 2k train / 1k val images from full dataset

We then used **GradCAM++** to visualize attention:
- For real dogs: focused on **nose, ears, and fur texture**
- For AI images: attention was **scattered**, often irrelevant or misaligned

#### Figure: Grad-CAM Heatmaps Revealing Model Attention Across Real and AI-Generated Dog Images
<img src="https://github.com/user-attachments/assets/6fdd6c85-adb7-456d-af1a-0792cb0cf90b" width="400"/>

These insights led us to test **snout-focused edge enhancement** using RGBA input. The model maintained ~95% accuracy—indicating CNNs already learn edge features well.

Next, we developed a **dual-branch model** (full image + cropped leg) to emphasize **leg structure**, achieving:
- 100% train / 98% validation accuracy on random data  
- But signs of **overfitting** emerged due to small sample size and model complexity

Regularization was introduced (dropout), but validation accuracy dropped to ~86% and became unstable.

**Summary:** Despite CNNs’ ability to localize important visual cues, their performance was consistently below our optimized **Bayesian MLP**, which reached up to **99% validation accuracy**. Additionally, CNN models are **30–50× slower** than MLPs and **require GPUs to run**, making them impractical for lightweight deployment. While leg-focused fusion models are conceptually promising, they are highly prone to overfitting and computationally expensive.


## Enhanced MLP with Leg-Based Feature (Final Model)
`Enhanced_Bayesian_MLP.ipynb`

Due to the inconsistent performance of our CNN models on randomly sampled data (~86% validation accuracy), we suspected that our **model capacity might be too large** for the **limited 2,000-image dataset**. To test this, we trained multiple CNN variants on the **full dataset** using GPU acceleration.

We explored three end-to-end CNNs, including one with **4-channel RGBA input**, where the fourth channel represents **edge information extracted from the image**. This edge-aware input helps CNNs emphasize object boundaries (e.g., legs, facial structure) that often appear distorted in AI-generated images. However, all three models exhibited **high variance in validation accuracy** (large performance fluctuations between epochs), likely due to:
- Noisy or redundant features from full images
- Difficulty in aligning edge maps with RGB signals
- Overfitting risk from excessive parameters vs. signal quality

#### Figure: Training and Validation Accuracy of the 4-Channel RGBA CNN Model
<img src="https://github.com/user-attachments/assets/6b97b434-1d6f-4dc3-b2c9-e0e6eb5b0636" width="300"/>

Given these challenges, we reverted to our **Bayesian MLP baseline** and focused on **augmenting it with leg-specific information**. Specifically, we:
- Used YOLOv5 to detect the dog’s bounding box
- Calculated the **grayscale brightness difference** between the full image and the leg region
- Appended this scalar as an **additional input feature**

This lightweight enhancement yielded strong results:
- Validation accuracy: **99.13%**
- ~30 more correctly classified samples compared to the vanilla MLP
- Grad-CAM visualizations showed improved sensitivity to **subtle leg structures, facial features, and expression cues**

#### Figure: Original Images (Left) and Salient Feature Focus Captured by the Enhanced Model (Right)
<img src="https://github.com/user-attachments/assets/7c8b3ee9-ecbb-4a5e-983f-01410897a639" width="440" height="700"/>
<img src="https://github.com/user-attachments/assets/86f0d1f3-d765-49de-adef-6480dc475219" width="440" height="700"/> 

**Conclusion:** While deep CNNs were powerful but unstable, our final MLP with a leg-aware scalar feature struck the best balance between interpretability, efficiency, and accuracy.


## Widget
`classify_new_image`

To test new images, we created a simple file upload widget (ipywidgets.FileUpload) to enable drag-and-drop or select an image from your device. After uploading, the image is passed through a feature extraction pipeline, and classified using the trained model (baseline_model.pt or plusdiff_model.pt).

The result is displayed with:
   - A predicted label: "Real 🐶" or "AI-generated 🤖"
   - A confidence score (between 0.00 and 1.00)

<img width="615" alt="Screenshot 2025-05-27 at 6 17 41 PM" src="https://github.com/user-attachments/assets/34787372-dc11-42a4-b1fc-86b0d6bc52f2" />


<img width="337" alt="Screenshot 2025-05-27 at 6 17 57 PM" src="https://github.com/user-attachments/assets/cea637f2-4f30-4423-bdc5-e09638f4af00" />

# Results
`baseline_model.pt`
`plusdiff_model.pt`

Finally, our two main models were saved as baseline_model.pt and plusdiff_model.pt. 

The baseline model was trained on the original RGB features, while the plusdiff model used the edge-enhanced RGBA features and a Bayesian architecture.

The baseline model worked fine, but felt a little limited in its ability to spot fakes. 

The plusdiff model was the clear winner—it combined smart inputs with uncertainty modeling, and it performed best across all metrics. This is the model we would deploy or build a UI around, especially since it offers both prediction and confidence.

| **Model Name**      | **Feature Set**                       | **Accuracy** | **Precision** | **Recall** | **Comments**                                                                        |
| ------------------- | ------------------------------------- | ------------ | ------------- | ---------- | ----------------------------------------------------------------------------------- |
| `baseline_model.pt` | CLIP embeddings                       | 98.3%        | 98.2%         | 98.1%      | Performs well on standard features, slightly lower generalization on edge cases.    |
| `plusdiff_model.pt` | CLIP embeddings + RGBA edge detection | 99.1%        | 99.0%         | 99.0%      | Best performance overall. Edge detection helps improve AI image detection accuracy. |

#### Figure: Accuracy Comparison between Baseline Model and Enhanced Model
![image](https://github.com/user-attachments/assets/1667047e-1716-4538-907e-4f735e145ed3)


# Final Takeaways and Next Steps
## Key Insights

   - The plusdiff model, which combines CLIP embeddings with RGBA edge detection, showed the strongest performance with over 99% accuracy, precision, and recall.
   - Adding edge-based features helped the model better detect subtle patterns often missed in AI-generated images, especially around fur texture and outlines.
   - The baseline model still performed well, but struggled slightly more with edge cases and synthetic artifacts.

## What's Next?
   - Expand the dataset to introduce more variety in dog breeds, image resolutions, and generative models to improve robustness.


