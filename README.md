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
4. [Results](#results)

# Problem Statement

AI-generated content is getting eerily real, especially with the recent release of Google Veo3, which can now generate hyper-realistic videos. 

This project aims to classify real dog images vs. AI-generated dog images using a combination of deep learning feature extraction, enhanced input processing, and Bayesian modeling to incorporate uncertainty in predictions. Specifically, our goals include:

- Build a reliable classifier to distinguish between real and AI-generated dog images.
- Enhance model performance through feature engineering (e.g., edge channel).
- Use Bayesian methods for uncertainity-aware predictions.
- Evaluate multiple models (CNN baseline, MLP variants with basic and engineered features) using consistent feature sets.


# Data and Repository Structure 
## Data
<a href='[https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset](https://www.kaggle.com/datasets/albertobircoci/ai-generated-dogs-jpg-vs-real-dogs-jpg)'>Our Kaggle dataset</a> contains a directory with real and AII generated dog images. The data is organized into three main subsets: Train, Validation, and Test. 

Each subset contains two subdirectories: Images and Labels. The Labels folder indicates the origin of each image, with 0 for real dog images and 1 for AI-generated ones. 

## Repository Structure
Our GitHub repository is organized as follows:

| Notebook/File                 | Purpose                                      |
|------------------------------|----------------------------------------------|
| `data_cleaning.ipynb`        | Load and preprocess image data               |
| `enhanced_bayesian_mlp.ipynb`| Add edge channel to images (RGBA)            |
| `cnn_method_exploration.ipynb`| CNN experiments using ResNet50              |
| `train_bayesian_mlp.ipynb`   | Train and evaluate Bayesian MLP              |
| `features_train_*.pt`        | Precomputed train feature tensors            |
| `baseline_model.pt`          | Trained model on base features               |
| `plusdiff_model.pt`          | Trained model on enhanced (plus) features    |



# Methodology

This project followed a structured yet flexible approach to solving the problem of classifying real vs. AI-generated dog images. We explored multiple model types and feature engineering techniques to find the best balance between accuracy, speed, and interpretability.

## Environment setup

We use Hugging Face's transformers library to import an image feature extractor and model backbone. This allows for efficient transfer learning and avoids training a deep CNN from scratch.

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


## Add edge channel to images 
`Enhanced_Bayesian_MLP.ipynb`

Next, we introduced edge detection to the images. Using Canny filters, we created a fourth channel that highlights edges—then combined it with the original RGB to form RGBA inputs. This was based on the idea that AI-generated images often have weird transitions or fuzzy outlines, and edge maps can help surface that.

   - Converted RGB images to grayscale, then added a Canny edge map as a 4th channel
   - Saved new image features as .pt tensors for fast reuse later
   - Models trained on these “plus” features saw a ~6–7% bump in accuracy, especially when distinguishing subtle fakes


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
   - The Bayesian MLP added a bit of robustness and interpretability, but the big leap came when we fed it the enhanced (RGBA) features
   - With the plus features, the Bayesian MLP hit ~86% accuracy and gave us confidence scores for each prediction
   - This made it easier to spot borderline cases or flag predictions the model wasn’t sure about

## Precompute feature tensors 
`features_train_*.pt`
`features_val_*.pt `
`features_test_*.pt`

We saved all our features as .pt files to speed things up during experimentation. The features_train_full.pt, features_val_full.pt, and features_test_full.pt files contained the original CNN-based embeddings from RGB images. 

The features_train_plus.pt, features_val_plus.pt, and features_test_plus.pt files held the same embeddings but with the edge channel added.

   - These precomputed tensors let us train MLPs in seconds instead of minutes
   - The “plus” versions were consistently more effective than the originals
   - We also made “cutted” versions of these files with smaller samples to use for quick debugging


# Results
`baseline_model.pt`
`plusdiff_model.pt`

Finally, our two main models were saved as baseline_model.pt and plusdiff_model.pt. 

The baseline model was trained on the original RGB features, while the plusdiff model used the edge-enhanced RGBA features and a Bayesian architecture.

The baseline model worked fine, but felt a little limited in its ability to spot fakes. 

The plusdiff model was the clear winner—it combined smart inputs with uncertainty modeling, and it performed best across all metrics. This is the model we would deploy or build a UI around, especially since it offers both prediction and confidence.

| Model Variant | Feature Set | Accuracy | Precision | Recall  | Comments                   |
| ------------- | ----------- | -------- | --------- | ------- | -------------------------- |
| Baseline MLP  | `*_full.pt` | \~79%    | \~78%     | \~76%   | Basic performance          |
| Bayesian MLP  | `*_full.pt` | \~80%    | \~79%     | \~77%   | Slight boost + uncertainty |
| **Bayesian MLP**  | **`*_plus.pt`** | **86%**  | **85%**   | **87%** | **Best overall performance**   |


