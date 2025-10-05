# ChestX-ray
Chest X-ray Multi-label Disease Classification
Overview
This project implements a deep learning pipeline for automated chest X-ray multi-label classification based on the NIH Chest X-ray Dataset. The aim is to classify 14 thoracic diseases from radiographs using state-of-the-art convolutional neural network (CNN) techniques. The pipeline leverages transfer learning, robust data preprocessing, class balancing, and interpretability tools such as Grad-CAM, facilitating reliable prediction and transparent insight into model decisions.

Dataset
Source: NIH Chest X-ray Dataset (Kaggle)

Size: ~112,000 images, 14 disease labels, single or multi-label per image

Files Used:

Chest X-ray images (PNG/JPG)

DataEntry2017.csv for image-label mapping

BBoxList2017.csv for bounding box information (optional)

Labels:

Atelectasis, Cardiomegaly, Effusion, Infiltration,

Mass, Nodule, Pneumonia, Pneumothorax,

Consolidation, Edema, Emphysema, Fibrosis,

Pleural Thickening, Hernia

License: CC0-1.0 (Open Data License)

Project Pipeline
1. Data Preparation
Extraction: Dataset is downloaded from Kaggle using authenticated API.

Organization: Images are recursively scanned and indexed; label CSV files are joined to image paths.

Splitting: Unique patient IDs are used to split data into train, validation, and test sets (ensuring no patient leakage).

Preprocessing:

Image resizing to 
224
×
224
224×224 pixels

Normalization of pixel values

Label encoding (multi-hot vectors for each image)

Data balanced with computed class weights to mitigate severe class imbalance.

2. Data Augmentation
Applied to training images (random horizontal flips, rotations, brightness variation) to increase model robustness.

3. Model Architecture
Backbone: DenseNet121 pretrained on ImageNet, excluding the top layer.

Custom Head:

Global Average Pooling

Dropout (0.5)

Dense output layer (14 nodes, sigmoid activation for multilabel)

Optimization: Adam optimizer (
1
×
10
−
4
1×10 
−4
  learning rate).

Loss Function: Custom Focal Loss to address label imbalance more effectively than binary cross-entropy.

Training:

ModelCheckpoint and EarlyStopping callbacks monitor validation loss to select best weights and prevent overfitting.

4. Model Evaluation
Metrics:

Training and validation accuracy

Binary cross-entropy loss

Disease-wise precision/recall supported by label-wise analysis.

Visualization:

Class distribution histogram

Training curves (accuracy, loss vs. epoch)

Grad-CAM heatmaps to highlight regions influencing specific disease predictions.

5. Interpretation & Visualization
Grad-CAM:

Visualizes learned spatial features for each disease label.

Useful for verifying model focus, supporting clinical trust.

Class Weights:

Calculated for each label using sklearn.utils.class_weight to address data imbalance.

6. Results
Sample Results:

Model achieves improvement in validation accuracy and balanced performance across disease classes.

Training stability enhanced using Focal Loss and class-balanced sampling.

Grad-CAM demonstrates model sensitivity to radiological disease features.

7. Limitations and Future Work
Imbalanced dataset for rare disease classes remains challenging despite class balancing.

Possible improvements include more advanced augmentation, ensemble models, or additional data sources.

Clinical validation required before deployment in real-world settings.

Linkdein -https://www.linkedin.com/in/rugved-bairagi-7882b5285/
