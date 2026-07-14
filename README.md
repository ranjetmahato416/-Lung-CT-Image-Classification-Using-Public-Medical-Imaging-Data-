# Lung Cancer Nodule Classification from Chest CT Images using Deep Learning



## Overview



This repository contains the complete implementation of my MSc Computer Science dissertation at the **University of East London**.



The project develops an end-to-end deep learning pipeline for **lung cancer nodule classification** using the publicly available **LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)** dataset.



Unlike many existing implementations that use pre-processed datasets, this project builds the complete preprocessing pipeline directly from the original DICOM CT scans and XML radiologist annotations. The pipeline automatically extracts lung nodules, generates consensus segmentation masks from multiple radiologists, standardizes the dataset, and prepares images for convolutional neural network (CNN) training.



---



# Research Objectives



The objectives of this project are to:



* Build a fully reproducible preprocessing pipeline from the original LIDC-IDRI dataset.

* Extract annotated pulmonary nodules using consensus annotations from multiple radiologists.

* Generate standardized CNN-ready image datasets.

* Train and evaluate deep learning models for benign and malignant lung nodule classification.

* Compare the performance of multiple CNN architectures.



---



# Dataset



**Dataset:** LIDC-IDRI (Lung Image Database Consortium)



The dataset contains:



* Thoracic CT scans in DICOM format

* XML annotations from up to four experienced thoracic radiologists

* Nodule segmentation contours

* Radiological characteristics

* Malignancy ratings



This project uses the original CT images together with the XML annotations to automatically construct the training dataset.



---



# Project Pipeline



The complete workflow consists of the following stages:



```

Raw DICOM CT Images

&#x20;         │

&#x20;         ▼

XML Annotation Parsing

&#x20;         │

&#x20;         ▼

Consensus Nodule Generation

&#x20;         │

&#x20;         ▼

3D Nodule Extraction

&#x20;         │

&#x20;         ▼

Consensus Segmentation Masks

&#x20;         │

&#x20;         ▼

Patch Standardization

&#x20;         │

&#x20;         ▼

128 × 128 CNN Dataset

&#x20;         │

&#x20;         ▼

Deep Learning Model Training

&#x20;         │

&#x20;         ▼

Performance Evaluation

```



---



# Repository Structure



```

Dissertation/



│

├── Notebooks/

│   ├── 01_Dataset_Exploration.ipynb

│   ├── 02_XML_Annotation_Parsing.ipynb

│   ├── 03_Metadata_Generation.ipynb

│   ├── 04_ROI_Extraction_Legacy.ipynb

│   ├── 05_pyLIDC_Exploration.ipynb

│   ├── 06_Nodule_Extraction.ipynb

│   ├── 07_Dataset_Standardization.ipynb

│   └── 08_Model_Training.ipynb (In Progress)

│

├── Dataset/

│   ├── Raw/

│   ├── ANNOTATION_STAGING/

│   └── Processed/

│

├── README.md

├── requirements.txt

└── .gitignore

```



---



# Current Progress



* Dataset exploration completed

* XML annotation parser implemented

* Metadata generation completed

* pyLIDC integration completed

* Consensus nodule extraction pipeline completed

* Automatic mask generation implemented

* Dataset checkpointing and resume support implemented

* CNN dataset generation pipeline implemented

* Deep learning model training (In Progress)



---



# Technologies Used



* Python

* Google Colab

* NumPy

* Pandas

* Matplotlib

* OpenCV

* PyDICOM

* pyLIDC

* Scikit-image

* Scikit-learn

* TensorFlow / Keras



---



# Dataset Output



The preprocessing pipeline generates:



* 3D lung nodule patches

* Consensus segmentation masks

* Standardized PNG images for CNN training

* Metadata CSV files

* Processing logs with automatic checkpointing



---



# Future Work



* Train multiple CNN architectures

* Hyperparameter optimisation

* Performance comparison

* Explainable AI (Grad-CAM)

* Model evaluation using ROC, Precision-Recall and Confusion Matrix

* Web-based inference prototype



---



# Author



**Ranjeet Kumar Mahato**



MSc Computer Science



University of East London



---



# License



This repository is intended for academic and research purposes.



The LIDC-IDRI dataset is distributed separately under its own licensing terms and is not included in this repository.


