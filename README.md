🥔 Potato Disease Classification using Deep Learning
This project focuses on building a deep learning model to automatically classify potato plant leaf images into three categories: Early Blight, Late Blight, 
and Healthy. The goal is to assist farmers and agricultural professionals in diagnosing plant diseases efficiently and accurately through image-based predictions.

📊 Dataset

Source: https://www.kaggle.com/datasets/emmarex/plantdisease

Categories:

Early Blight

Late Blight

Healthy

Format: RGB images of potato leaves

🧠 Model Architecture

Base Model: Convolutional Neural Network (CNN)

Layers Used:

Convolutional layers (Conv2D)

MaxPooling

Batch Normalization

Dropout for regularization

Dense (fully connected) layers

Activation: ReLU and Softmax

Loss Function: Categorical Crossentropy

Optimizer: Adam

Metrics: Accuracy, Precision, Recall

🚀 Results

Training Accuracy: ~98%

Validation Accuracy: ~95%

Test Accuracy: ~94%

Confusion matrix and classification report were used to assess performance on each class.

📌 Key Insights

Data augmentation helped prevent overfitting and improved generalization.

The model successfully distinguished between Early and Late Blight with high precision.

The CNN was able to generalize well despite the small dataset size due to careful preprocessing and regularization.

🛠️ Tools & Libraries

Python

TensorFlow / Keras

NumPy

OpenCV

Matplotlib

Scikit-learn

