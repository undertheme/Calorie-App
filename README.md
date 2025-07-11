# 🥗 Calorie App: AI-Powered Calorie Estimation from Food Images
CalorieAI is an AI-powered app that estimates the calories of food from images using state-of-the-art vision-language models and Retrieval-Augmented Generation (RAG). Built on top of the Food-101 dataset, this project combines deep learning, LLMs, and nutritional science into a complete end-to-end pipeline.

---

## 🚀 Features

- **Image Recognition**: Utilizes a fine-tuned [LLaMA-3.2 Vision model] to identify food items from images.
- **Ingredient Extraction**: Predicts ingredients along with their quantities and units.
- **Calorie Calculation**: Employs a RAG setup to fetch nutritional information and compute total calorie content.
- **User-Friendly Interface**: Provides an intuitive interface for users to upload images and receive instant calorie estimations.

---

## 📊 Dataset

The model is trained on a customized version of the [Food-101 dataset](https://www.kaggle.com/datasets/dansbecker/food-101), enriched with ingredient and quantity annotations.

---

## 🧠 Model

The core of the application is a fine-tuned LLaMA-3.2 Vision model, optimized using LoRA for efficient training.

---

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/undertheme/Calorie-App.git
   cd Calorie-App
