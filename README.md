# Neural Network Design and Deployment: MNIST to Fashion-MNIST

This repository contains Google Colab notebooks developed for **Practical No. 01: Neural Network Design and Deployment**. The project demonstrates the lifecycle of building a deep learning model, starting from a simple Multi-Layer Perceptron (MLP) for digit recognition and advancing to a Convolutional Neural Network (CNN) for fashion item classification.

## 🚀 Project Overview
The objective of this assignment was to:
* **Model Development:** Design, train, and optimize a neural network.
* **Architecture Selection:** Transition from MLP to CNN for image data.
* **Data Preprocessing:** Implement normalization, reshaping, and one-hot encoding.
* **Evaluation:** Use loss functions (Categorical Crossentropy) and optimizers (Adam) to reach high accuracy.
* **Deployment:** Export the trained model for production use.

## 📂 Repository Structure
* `Untitled97.ipynb`: The initial implementation using the standard MNIST Digits dataset with an MLP architecture (Accuracy: ~97.82%).
* `ASSIGNMENT_NO_1.ipynb`: The final assignment notebook using the Fashion-MNIST dataset with an optimized CNN architecture (Accuracy: ~91%).
* `fashion_model_v1.h5`: The serialized Keras model ready for deployment.

## 🛠️ Tech Stack
* **Platform:** Google Colab
* **Hardware:** NVIDIA T4 GPU (for CNN acceleration)
* **Framework:** TensorFlow / Keras
* **Libraries:** NumPy, Matplotlib, OS


## 🧠 Model Evolution

### Phase 1: MLP (Digits)
* **Dataset:** 28x28 Grayscale digits (0-9).
* **Layers:** Flatten -> Dense (200) -> Dense (150) -> Softmax (10).
* **Why:** Good for simple pattern recognition but lacks spatial awareness.

### Phase 2: CNN (Fashion)
* **Dataset:** Fashion-MNIST (T-shirts, Shoes, Bags, etc.).
* **Architecture:**
    * **Conv2D (32 filters):** Detects edges and textures.
    * **BatchNormalization:** Stabilizes training and speeds up convergence.
    * **MaxPooling2D:** Reduces spatial dimensions.
    * **Dropout (0.3):** Prevents overfitting by randomly deactivating neurons.
* **Why CNN?** Convolutional layers are superior for images as they extract local features and preserve spatial relationships, whereas MLPs flatten images and lose structural context.

## 📊 Results

| Dataset | Architecture | Epochs | Accuracy |
| :--- | :--- | :--- | :--- |
| MNIST Digits | MLP | 10 | 97.82% |
| Fashion-MNIST | CNN | 15 | 90.60% |

## 💻 How to Run
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the `.ipynb` files.
3. Go to **Runtime** > **Change runtime type** and select **GPU**.
4. Run all cells to train the model and generate the prediction visualizations.

## 📁 Deployment
The model is exported as an HDF5 file (`.h5`). You can load this model in a Python environment using:

```python
from tensorflow.keras.models import load_model
model = load_model('fashion_model_v1.h5')
