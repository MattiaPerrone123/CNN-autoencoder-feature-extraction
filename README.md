# CNN Autoencoder Feature Extraction

# Medical Image Analysis
This repository contains the codebase for a project focused on feature extraction from medical images using a convolutional neural network (CNN)-based autoencoder. The project explores the use of autoencoders to identify latent features in segmented medical images, with applications in analyzing spinal morphology and associated clinical conditions. Initial findings demonstrate the potential of CNN autoencoders for analyzing complex medical imaging datasets. For more details, please see the [abstract](docs/abstract.pdf) (poster at ORS 2025 annual meeting).

## Data
The input data consists of medical images, including both structural (e.g., CT scans) and functional (e.g., PET scans) modalities. These data are not included in the repository but follow the directory structure outlined in the `data/` folder. Subfolders contain images and masks for individual patients. Example input data includes:

- Structural images: CT or MRI scans stored as `.mha` files
- Functional images: PET scans for advanced analysis
- Annotations: Ground truth values provided in `annotations.xlsx`

These files are processed through the pipeline, which integrates segmentation, preprocessing, and feature extraction.


!!! Metto immagine pipeline da qualche parte!!!

## Key Features
- **Segmentation of Images**:
  - Segmentation of spinal regions in 3D medical images for downstream analysis.
- **Feature Extraction**:
  - Extracting latent features using a CNN-based autoencoder for pattern identification in medical data.
- **Geometry Computation and Analysis**:
  - Computing geometrical parameters of latent features and correlating them with clinical labels.
- **Evaluation**:
  - Comprehensive evaluation using metrics such as reconstruction loss and correlation with ground truth annotations.

## Repository Structure

### Root Directory
- **data/**: Placeholder for input data (not included in the repository), including:
  - **annotations.xlsx**: Ground truth annotations for the dataset.
  - **images/**: Input images (e.g., `.mha` files).
  - **masks/**: Corresponding masks for the input images.
- **docs/**: Documentation and resources, including:
  - **abstract.pdf**: Abstract presented at the ORS 2025 annual meeting.
  - **poster.pptx**: Poster presentation slides.
  - **podium.pptx**: Slides for a podium presentation.
- **main.ipynb**: Jupyter Notebook demonstrating the pipeline's usage.
- **requirements.txt**: Python dependencies required for the project.
- **.gitignore**: Specifies files and folders to exclude from version control.

The **src/** directory contains all scripts and modules for processing, analysis, and utility functions:
- **autoencoder.py**: Definition of the CNN-based autoencoder model.
- **compute_geometry.py**: Geometry computations on the latent space.
- **config.py**: Stores configuration constants and parameters.
- **data_loader.py**: Data loading utilities for handling input datasets.
- **image_generation.py**: Functions for generating augmented images.
- **image_preprocessing.py**: Preprocessing pipelines for medical images.
- **metrics.py**: Tools for evaluating model performance and reconstruction.
- **model_training.py**: Training pipeline for the autoencoder model.
- **utils.py**: General-purpose helper functions.

