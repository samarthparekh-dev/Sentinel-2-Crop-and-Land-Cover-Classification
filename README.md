# Sentinel-2 Crop & Land Cover Classification (EuroSAT)

An end-to-end Earth Observation (EO) Computer Vision pipeline that classifies multi-spectral Sentinel-2 satellite imagery into 10 distinct agricultural and land cover categories.

This project goes beyond basic image classification by implementing Transfer Learning (ResNet-18), robust training guardrails (Early Stopping & LR Scheduling), and Explainable AI (Saliency Maps) to ensure model transparency and prevent black-box predictions.

Key Features

1) Authentic EO Dataset: Utilizes the EuroSAT dataset, containing 27,000 real Sentinel-2 satellite image patches.

2) Transfer Learning: Fine-tunes a pre-trained ResNet-18 architecture, adapting its feature-extraction capabilities from standard ImageNet to complex geospatial spectral patterns.

3) Explainable AI (XAI): Implements backpropagated Saliency Maps to visually highlight the exact pixel regions the neural network focused on to make its predictions.

4) Training:

  -Automated Early Stopping to monitor validation loss and halt training before overfitting occurs.

  -ReduceLROnPlateau Scheduler to dynamically adjust the learning rate for optimal convergence.

  -Geographic/Spatial Data Augmentation (Random flips, Color Jittering).


Dataset & Classes : 

The model predicts across 10 classes, heavily focused on agriculture and foundational land cover:
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.
 Performance & Results

The pipeline achieves state-of-the-art performance for this architecture:

Overall Accuracy: ~95%

Macro F1-Score: 0.95

Evaluation Metrics: Outputs a full sklearn classification report (Precision, Recall, F1) and a Seaborn-generated Confusion Matrix to track misclassifications between overlapping classes (e.g., Highway vs. River).

Explainable AI Output

Instead of blindly trusting the model's output, the Saliency Map generator proves why the model made its decision by projecting a heat map over the regions of maximum gradient attention.

 Installation & Usage

1. Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

    pip install torch torchvision numpy scikit-learn matplotlib seaborn


2. Run the Pipeline

The script will automatically download the 27,000-image EuroSAT dataset (if not locally present), initialize the ResNet model, train it, and generate the visualizations.

    python classification.py


Architecture Overview :

Data Preprocessing: Images are converted to PyTorch tensors and normalized to standard ImageNet parameters.

Model: A frozen ResNet-18 backbone with an unfrozen layer4 and a custom fully-connected classification head tailored to 10 classes.

Loss & Optimizer: CrossEntropyLoss and the Adam Optimizer (1e-3 learning rate).

Validation: 80/20 Train-Test split, ensuring the model's metrics are generated purely on unseen geospatial data.

Developed for Earth Observation and AgTech Computer Vision applications.


RESULTS : 

--- FINAL CLASSIFICATION REPORT ---
                            
                            precision    recall  f1-score   support

              AnnualCrop       0.94      0.97      0.96       589
                  Forest       0.98      0.99      0.99       597
    HerbaceousVegetation       0.91      0.94      0.93       568
                 Highway       0.94      0.90      0.92       502
              Industrial       0.97      0.97      0.97       536
                 Pasture       0.98      0.89      0.93       374
           PermanentCrop       0.91      0.92      0.92       498
             Residential       0.97      0.98      0.97       613
                   River       0.93      0.95      0.94       483
                 SeaLake       1.00      0.98      0.99       640

                accuracy                           0.95      5400
               macro avg       0.95      0.95      0.95      5400
            weighted avg       0.95      0.95      0.95      5400



<img width="938" height="790" alt="image" src="https://github.com/user-attachments/assets/b1ac5899-dcfa-4658-a03e-f048b14fdd5b" />
<img width="1490" height="466" alt="image" src="https://github.com/user-attachments/assets/34b3da61-46c0-4f09-b77b-32031355a392" />
<img width="1447" height="593" alt="image" src="https://github.com/user-attachments/assets/718fa78a-216f-423f-a0c2-eb02c1c425d0" />
