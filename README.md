🧬 Overview
This project presents a multimodal machine learning system to diagnose Polycystic Ovary Syndrome (PCOS) using both clinical blood report data and ultrasound images.
By combining structured (tabular) and unstructured (image) data sources, the system aims to improve early detection accuracy using classical machine learning and deep learning approaches.

🚀 Features

🔬 Tabular Diagnosis:Uses features such as BMI, testosterone levels, and follicle counts.

🖼️ Image Diagnosis: CNN model trained on ultrasound images to detect ovarian morphology.

🔗 Multimodal Fusion: A soft fusion technique combines outputs from both models.

🌐 Interactive Dashboard: Streamlit-based frontend allows clinicians to input values and upload images for diagnosis.

📊 Data Sources
Tabular Data: Extracted and partially synthetically augmented from Kaggle datasets (~3000 rows).

Image Data: Ultrasound images categorized into PCOS and Non-PCOS.

🧠 Model Details
1. Tabular Classifiers:
Logistic Regression

Random Forest (Best Performer)

Support Vector Machine

2. Image Classifier:
CNN (Convolutional Neural Network):

Input: 224x224 grayscale images

Architecture: 3 Conv blocks → Dense layers → Sigmoid

Regularization: Dropout (0.5), EarlyStopping

Optimizer: Adam

📈 Performance
Random Forest:

F1-score: 0.95 (Class 1)

CNN:

AUC: ~0.90+

Fusion:

Increased diagnostic stability and accuracy

📸 Streamlit Interface
Inputs:

Numerical: BMI, Testosterone, AFC

Image: Ultrasound scan (.png, .jpg, .jpeg)

Output:

PCOS Diagnosis probability

Visual confirmation

🔮 Future Enhancements
Meta-model fusion strategies

Explainability using Grad-CAM, SHAP

Real-time clinical integration

Mobile & Edge deployment with TensorFlow Lite

Inclusion of time-series hormone data
