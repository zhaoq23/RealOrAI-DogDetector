# Bark or Bot?
## Detecting Real vs. AI Dog Images

<img src="https://i.ytimg.com/vi/b7rB7VFl_I4/maxresdefault.jpg" alt="Dog vs Robot" width="400"/>

# Table of Contents
1. [Problem Statement](#problem-statement)
2. [Data](#data)
3. [Methodology](#methodology)<br>
	3.1. [Required Packages](#required-packages)<br>
	3.2. [Load and Preprocess Data](#load-and-preprocess-data)<br>
	3.3. [Load Pre-Trained Vision Transformer (ViT) and Extract Features](#load-pre-trained-vision-transformer-vit-and-extract-features)<br>
	3.4. [Train-Test Split and DataLoader Creation](#train-test-split-and-dataloader-creation)<br>
	3.5. [Bayesian Neural Network for Classification](#bayesian-neural-network-for-classification)<br>
	3.6. [Model Training](#model-training)<br>
	3.7. [Model Evaluation](#model-evaluation)<br>
	3.8. [Bayesian Inference for Uncertainty Estimation](#bayesian-inference-for-uncertainty-estimation)<br>
	3.9. [Feature Importance Analysis Using Gradients](#feature-importance-analysis-using-gradients)<br>
	3.10. [Visualizing Feature Importance on the Image](#visualizing-feature-importance-on-the-image)<br>
	3.11. [Occlusion Sensitivity](#occlusion-sensitivity)
4. [Classifying real-world images](#classifying-real-world-images)<br>
5.1. [Results](#results)

# Problem Statement

AI-generated content is getting eerily real, especially with last week’s release of Google Veo, which can now generate hyper-realistic videos. 

This project uses Bayesian modeling to sniff out whether a dog image is real or AI-generated. The goal is to build a model that doesn’t just predict—but shows us how sure it is, and why.

# Data
<a href='[https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset](https://www.kaggle.com/datasets/albertobircoci/ai-generated-dogs-jpg-vs-real-dogs-jpg)'>Kaggle dataset</a> contains a directory with real and AII generated dog images. The data is organized into three main subsets: Train, Validation, and Test. 

Each subset contains two subdirectories: Images and Labels. The Labels folder indicates the origin of each image, with 0 for real dog images and 1 for AI-generated ones. 

# Methodology

enter here

# Results

enter here
