# CNN Autoencoder Feature Extraction

# Medical Image Analysis
This repository contains the codebase for a project focused on feature extraction from medical images using a convolutional neural network (CNN)-based autoencoder. The project explores the use of autoencoders to identify latent features in segmented medical images, with applications in analyzing spinal morphology and associated clinical conditions. Initial findings demonstrate the potential of CNN autoencoders for analyzing complex medical imaging datasets. For more details, please see the [abstract](docs/abstract.pdf) (poster at ORS 2025 annual meeting).

## Data
The input data consists of medical images, including both structural (e.g., CT scans) and functional (e.g., PET scans) modalities. These data are not included in the repository but follow the directory structure outlined in the `data/` folder. Subfolders contain images and masks for individual patients. Example input data includes:

- Structural images: CT or MRI scans stored as `.mha` files
- Functional images: PET scans for advanced analysis
- Annotations: Ground truth values provided in `annotations.xlsx`

These files are processed through the pipeline, which integrates segmentation, preprocessing, and feature extraction.




## Pipeline
- **Segmentation of MRI scans**:
  - Segmenting lumbar discs in MRI scans to isolate regions of interest for further analysis
- **Feature extraction**:
  - Extracting latent features from segmented MRI scans using a CNN-based autoencoder 
- **Geometric features computation**:
  - Computing disc geometry such as disc height, width and orientation
- **Disc narrowing prediction**:
  - Predicting disc narrowing by leveraging extracted features and correlating them with patient-specific clinical labels.
- **Feature interpretability**:
  - Analyzing the interpretability of autoencoder latent features by correlating them the extracted geometric features


<br>

<p align="center">
  <img src="figures/pipeline.png" width="1250" height="230">
</p>

<br>


## Repository Structure

### Root Directory
- ```data```: Placeholder for input data (not included in the repository), including:
  - **annotations.xlsx**: Ground truth annotations for the dataset
  - ```images```: Input MRI scans
  - ```masks```: Corresponding masks for the input MRI scans
- ```docs```: Documentation and resources, including:
  - **abstract.pdf**: Abstract presented at the ORS 2025 annual meeting
  - **poster.pptx**: Poster presentation slide (ORS 2025)
  - **podium.pptx**: Slides for podium presentation (PSRS 2024)
- **main.ipynb**: Jupyter Notebook demonstrating the pipeline's usage
- **requirements.txt**: Python dependencies required for the project
- **.gitignore**: Specifies files and folders to exclude from version control

The ```src``` directory contains all scripts and modules for processing, analysis, and utility functions:
- **autoencoder.py**: Definition of the CNN-based autoencoder model
- **compute_geometry.py**: Computations of relevant geometric features
- **config.py**: Stores configuration constants and parameters
- **data_loader.py**: Data loading utilities for handling input datasets
- **image_generation.py**: Functions for generating images from latent features
- **image_preprocessing.py**: Preprocessing functions 
- **metrics.py**: Tools for evaluating model performance
- **model_training.py**: Training pipeline for the ML models
- **utils.py**: General-purpose helper functions

