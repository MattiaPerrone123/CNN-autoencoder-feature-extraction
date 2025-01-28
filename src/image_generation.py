import torch
import numpy as np
import matplotlib.pyplot as plt
from .utils import calculate_feature_range, calculate_fixed_values


def generate_images(decoder, feature_index, feature_range, fixed_values, numb_plots):
    """Generate images by varying a single feature in the latent space"""
    images=[]
    latent_vector_feature=[]
    
    if fixed_values.dim()==1:
        fixed_values=fixed_values.unsqueeze(0)
    
    for value in np.linspace(feature_range[0], feature_range[1], num=numb_plots):
        latent_vector=fixed_values.clone()
        latent_vector[0, feature_index]=value  
        
        with torch.no_grad():
            generated_image=decoder(latent_vector).cpu().numpy()
        images.append(generated_image)
        latent_vector_feature.append(latent_vector)
    
    return images, latent_vector_feature


def visualize_generated_images(images, numb_plots, view="sagittal"):
    """Visualize generated images for a single feature."""
    fig, axes=plt.subplots(1, numb_plots, figsize=(numb_plots * 2, 2))
    for i, image in enumerate(images):
        if view=="sagittal":
            axes[i].imshow(image[0, 0, 32], cmap='gray') 
        elif view=="coronal":
            axes[i].imshow(np.rot90(image[0, 0, :, 32], k=3), cmap='gray')  
        else:
            raise ValueError("Invalid view option. Choose 'slice' or 'rotated'.")
        axes[i].axis('off')
    plt.show()


def flatten_latent_features(latent_features_list, latent_dim, numb_plots):
    """Flatten the list of latent feature tensors"""
    latent_features_flat = [
        latent_features_list[i][j]
        for i in range(latent_dim)
        for j in range(numb_plots)
    ]
    return torch.cat(latent_features_flat, dim=0)


def process_images_list(images_list, threshold=0.5, shape=(-1, 64, 64, 64)):
    """Convert a list of images into a binary array and reshape it"""
    images_arr=(np.squeeze(images_list) > threshold).astype(int)
    images_arr=images_arr.reshape(shape)
    
    return images_arr

def process_latent_features_and_generate_images(latent_representations, 
    decoder, 
    numb_plots=6, 
    latent_dim=4, 
    view="sagittal"  
):
    """Process latent features, generate images, and return results."""
    images_list = []
    latent_features_list = []

    decoder.to("cpu")

    for feature_index in range(latent_dim):
        feature_range = calculate_feature_range(
            latent_representations,  
            feature_index
        )
        fixed_values = calculate_fixed_values(latent_representations)

        images, latent_vector_feature = generate_images(
            decoder=decoder,
            feature_index=feature_index,
            feature_range=feature_range,
            fixed_values=fixed_values,
            numb_plots=numb_plots
        )

        images_list.append(images)
        latent_features_list.append(latent_vector_feature)

        visualize_generated_images(images, numb_plots, view=view)

    images_arr=process_images_list(images_list, threshold=0.5, shape=(-1, 64, 64, 64))

    latent_features_list_fin=flatten_latent_features(
        latent_features_list, latent_dim, numb_plots
    )

    return images_arr, latent_features_list_fin
