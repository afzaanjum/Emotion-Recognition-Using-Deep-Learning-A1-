# Emotion-Recognition-Using-Deep-Learning-A1-
Deep learning project for emotion recognition using both classification and regression approaches. Includes experiments with EfficientNetV2B0, MobileNetV2, and XceptionNet for categorical emotion classification, and a VGG16-based regression model for predicting continuous valence–arousal values. 

Emotion Recognition using Deep Learning
📌 Project Overview

This project focuses on emotion recognition from images using deep learning approaches. Two complementary tasks were explored:

Categorical Emotion Classification – Predicting discrete emotion labels using transfer learning on EfficientNetV2B0, MobileNetV2, and XceptionNet.

Continuous Emotion Regression – Predicting Valence and Arousal values using a VGG16-based regression model.

The repository demonstrates transfer learning, fine-tuning, and evaluation using both classification metrics and continuous domain evaluation metrics.

📂 Dataset

Images were preprocessed and resized to 224×224×3.

Data split into Training / Validation / Test sets.

Labels:

Classification: Multiple emotion categories.

Regression: Continuous valence–arousal annotations.

🧠 Model Architectures
1. EfficientNetV2B0

Parameters: 6.2M total, ~330k trainable.

Fine-tuned with Adam (lr=1e-5).

Used EarlyStopping and ReduceLROnPlateau callbacks.

2. MobileNetV2

Lightweight CNN for mobile vision tasks.

Trained with Adam (lr=1e-3), batch size 128.

Callbacks applied for stability and early convergence.

3. XceptionNet

Deep CNN using depthwise separable convolutions.

Fine-tuned with Adam (lr=1e-3 → reduced on plateau).

4. Regression Model (VGG16 Backbone)

Pretrained VGG16 as base, frozen layers.

Added dense layers + dual outputs:

Valence (continuous)

Arousal (continuous)

Loss: MSE, Optimizer: Adam.

⚙️ Training Details

Batch size: 128

Epochs: 10–15 (with early stopping)

Callbacks:

EarlyStopping (patience=5, restore best weights)

ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)

Hardware: Trained on GPU runtime (e.g., Google Colab).

📊 Results
Classification Models
Model            |	Accuracy	|  F1-Score	|  AUC-ROC	|  Cohen’s Kappa	|  PR-AUC
EfficientNetV2B0  12.12%	    0.0420   	  0.5090  	-0.0030	        0.1392	
  MobileNetV2     36.36%	    0.3587	    0.7771	   0.2726	        0.3845	
XceptionNet       25.51%	    0.2464	    0.6946	   0.1486	        0.2612	

✅ MobileNetV2 outperformed other baselines in classification.

Regression Model (VGG16)
Metric	Valence	  Arousal
RMSE	   1.7424	  0.3960
CORR	   -0.0503	0.0623
SAGR	   0.2626	  0.7576
CCC	     -0.0020	0.0345

✅ SAGR indicates reasonable trend consistency, though correlation and CCC remain low.



🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/emotion-recognition.git
cd emotion-recognition


Install dependencies:

pip install -r requirements.txt


Train models:

python train_mobilenet.py
python train_effnet.py
python train_xception.py
python train_regression.py


Evaluate models:

python evaluate.py

📝 Discussion

Classification Models: MobileNetV2 showed the best performance, balancing accuracy and generalization.

Regression Model: While RMSE values were acceptable, low correlation and CCC highlight difficulty in predicting precise valence–arousal values.

For real-world applications, SAGR may be the most reliable metric, as it emphasizes correct trend direction rather than exact values.


📧 Contact
Email: afzaanjum1@gmail.com
