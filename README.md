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

# Data and Repository Structure 
## Data
<a href='[https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset](https://www.kaggle.com/datasets/albertobircoci/ai-generated-dogs-jpg-vs-real-dogs-jpg)'>Our Kaggle dataset</a> contains a directory with real and AII generated dog images. The data is organized into three main subsets: Train, Validation, and Test. 

Each subset contains two subdirectories: Images and Labels. The Labels folder indicates the origin of each image, with 0 for real dog images and 1 for AI-generated ones. 

## Repository Structure
Our GitHub repository is organized as follows:

![1748384438508](https://github.com/user-attachments/assets/d817e3c6-8e28-49dd-acd0-af1b32aad870)

RealOrAI-DogDetector/
├── data_clean.ipynb               # Preprocesses and cleans image data
├── Enhanced_Bayesian_MLP.ipynb    # Adds edge detection channel (RGBA) for better feature capture
├── method_explore.ipynb           # Experiments with CNNs (e.g., ResNet50) and compares architectures
├── Bayesian MLP Model_Train.ipynb # Trains a Bayesian MLP model with uncertainty estimation
├── features_train_*.pt            # Precomputed feature tensors for training
├── features_val_*.pt              # Precomputed feature tensors for validation
├── features_test_*.pt             # Precomputed feature tensors for testing
├── baseline_model.pt              # Trained baseline model using standard features
├── plusdiff_model.pt              # Trained enhanced model using edge-aware features
├── classify_new_image.ipynb       # Upload and classify your own dog image (real vs AI)
└── README.md                      # This file


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

## Simple MLP Model

We use a simple Multi-Layer Perceptron (MLP) with two hidden layers (128 and 64 units) and dropout for regularization. The model takes 384-dimensional feature vectors as input and predicts whether an image is real or AI-generated.

- **Input**: 384-d feature vector  
- **Architecture**: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Output
- **Validation Accuracy**: 98.98%  
- **Training Accuracy**: 98.72%  

This model is efficient and performs well on extracted image features.

## Add edge channel to images 
`Enhanced_Bayesian_MLP.ipynb`

Next, we introduced edge detection to the images. Using Canny filters, we created a fourth channel that highlights edges—then combined it with the original RGB to form RGBA inputs. This was based on the idea that AI-generated images often have weird transitions or fuzzy outlines, and edge maps can help surface that.

   - Converted RGB images to grayscale, then added a Canny edge map as a 4th channel
   - Saved new image features as .pt tensors for fast reuse later
   - Models trained on these “plus” features saw a ~6–7% bump in accuracy, especially when distinguishing subtle fakes

#### Figure: Grad-CAM Heatmaps Revealing Model Attention Across Real and AI-Generated Dog Images
![output](https://github.com/user-attachments/assets/6fdd6c85-adb7-456d-af1a-0792cb0cf90b)

   - The heatmaps illustrate the regions the model focused on when classifying dog images across four prediction types. The model consistently attends to key visual features such as the nose, eyes, ears, and fur texture—suggesting it relies on these areas to distinguish between real and AI-generated dogs, regardless of classification correctness.

## CNN model experiments 
`method_explore.ipynb`

To establish a benchmark, we fine-tuned a pretrained ResNet50. This was our strongest model out-of-the-box. The goal was to see how a CNN with deep spatial understanding would perform compared to the lightweight MLPs we planned to train later. We also used GradCAM++ to visualize where the network was “looking.”

   - ResNet50 hit ~87% accuracy on the validation set
   - GradCAM++ showed that for real dogs, attention centered on ears, snouts, and fur texture
   - For fake dogs, the heatmaps were scattered—often lighting up irrelevant background or oddly shaped limbs
   - These visuals confirmed that even CNNs were picking up on the awkwardness of AI images


## Train Bayesian MLP and evaluate
`Bayesian MLP Model_Train.ipynb`

Once we had features extracted from the images, we trained our classifiers. 

We compared two types: a standard MLP and a Bayesian version that uses dropout to estimate uncertainty. 

These models didn’t operate on raw images—they were trained on precomputed feature tensors (either from RGB images or the edge-enhanced RGBA ones).
   - The baseline MLP trained on regular RGB features achieved ~79% accuracy
   - The Bayesian MLP added a bit of robustness and interpretability ~80% accuracy
   - - The big leap came when we fed it the enhanced (RGBA) features. With the plus features, the Bayesian MLP hit ~86% accuracy and gave us confidence scores for each prediction
       
The Bayesian MLP (RGBA features) model made it easier to spot borderline cases or flag predictions the model wasn’t sure about.

## Precompute feature tensors 
`features_train_*.pt`
`features_val_*.pt `
`features_test_*.pt`

We saved all our features as .pt files to speed things up during experimentation. The features_train_full.pt, features_val_full.pt, and features_test_full.pt files contained the original CNN-based embeddings from RGB images. 

The features_train_plus.pt, features_val_plus.pt, and features_test_plus.pt files held the same embeddings but with the edge channel added.

   - These precomputed tensors let us train MLPs in seconds instead of minutes
   - The “plus” versions were consistently more effective than the originals
   - We also made “cutted” versions of these files with smaller samples to use for quick debugging

#### Figure: Original Images (Left) and Salient Feature Focus Captured by the Enhanced Model (Right)
![image](https://github.com/user-attachments/assets/7c8b3ee9-ecbb-4a5e-983f-01410897a639)
![image](https://github.com/user-attachments/assets/86f0d1f3-d765-49de-adef-6480dc475219)


## Widget
`classify_new_image`

To test new images, we created a simple file upload widget (ipywidgets.FileUpload) to enable drag-and-drop or select an image from your device. After uploading, the image is passed through a feature extraction pipeline, and classified using the trained model (baseline_model.pt or plusdiff_model.pt).

The result is displayed with:
   - A predicted label: "Real 🐶" or "AI-generated 🤖"
   - A confidence score (between 0.00 and 1.00)

#### Known Limitations 
   - Currently, uploaded images always return “Real 🐶” with 0.00 confidence, due to a mismatch between the uploaded image format and what the model expects.
   - The inference pipeline needs to match the training preprocessing exactly (including feature embedding and input shape).
   - Future iterations will address fixing input handling by ensuring that all uploaded images pass through the same extract_features() function used in training.

<img width="615" alt="Screenshot 2025-05-27 at 4 50 27 PM" src="https://github.com/user-attachments/assets/af881527-a286-461c-b304-8b982d0225df" />


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
   - Fix inference for uploaded images. Uploaded images currently return "Real" with 0 confidence due to preprocessing mismatches. Ensuring consistent feature extraction between training and inference should resolve this.
   - Expand the dataset to introduce more variety in dog breeds, image resolutions, and generative models to improve robustness.


