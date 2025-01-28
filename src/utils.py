import re
import torch
import os
import pandas as pd
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import get_dice, get_iou_train
from .image_preprocessing import process_image, process_mask


def extract_number(filename):
    """Extract the leading number from the filename"""
    match=re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def extract_number_from_string(input_string):
    """Search for a number in the input string and return it"""
    number_found=re.search(r'\d+', input_string)
    return number_found.group() if number_found else None


def get_file_paths(curr_path):
    """Generate paths for images, masks, and CSV files"""
    images_dir=os.path.join(curr_path, "arXiv", "images", "images")
    masks_dir=os.path.join(curr_path, "arXiv", "masks", "masks")
    rad_grad_path=os.path.join(curr_path, "radiological_gradings.csv")
    overview_path=os.path.join(curr_path, "overview.csv")
    return images_dir, masks_dir, rad_grad_path, overview_path


def load_tables(overview_path, rad_grad_path):
    """Load overview and radiological grading tables"""
    overview_table=pd.read_csv(overview_path)
    rad_grad_table=pd.read_csv(rad_grad_path)
    return overview_table, rad_grad_table


def load_and_sort_files(images_dir, masks_dir, extract_number):
    """Load and sort files from image and mask directories"""
    file_list_img=sorted(os.listdir(images_dir), key=extract_number)
    file_list_msk=sorted(os.listdir(masks_dir), key=extract_number)
    file_list_img_we=[filename.removesuffix(".mha") for filename in file_list_img]
    file_list_msk_we=[filename.removesuffix(".mha") for filename in file_list_msk]
    return file_list_img, file_list_msk, file_list_img_we, file_list_msk_we


def process_files(sorted_files_img, sorted_files_img_we, images_dir, masks_dir, overview_table, target_resolution):
    """Process image and mask files, segregate T1 and T2 data"""
    img_t1, img_t2=[], []
    msk_t1_disk, msk_t2_disk=[], []
    filename_t1, filename_t2=[], []
    num_disc_table_t1, num_disc_table_t2=[], []
    num_labels_real_t1_list, num_labels_real_t2_list=[], []

    for i, (img_file, img_file_we) in enumerate(zip(sorted_files_img, sorted_files_img_we)):
        img_path=os.path.join(images_dir, img_file)
        np_image=process_image(img_path, target_resolution)
        if np_image is None:
            continue

        msk_path=os.path.join(masks_dir, img_file)
        masks=process_mask(msk_path, np_image.shape)
        if i%10==0:
            print(i)

        if "t1" in img_path:
            append_data(img_t1, msk_t1_disk, filename_t1, num_disc_table_t1, num_labels_real_t1_list, np_image, masks, overview_table, img_file_we, "disk")
        elif "t2" in img_path:
            append_data(img_t2, msk_t2_disk, filename_t2, num_disc_table_t2, num_labels_real_t2_list, np_image, masks, overview_table, img_file_we, "disk")

    return img_t1, img_t2, msk_t1_disk, msk_t2_disk, filename_t1, filename_t2, num_disc_table_t1, num_disc_table_t2, num_labels_real_t1_list, num_labels_real_t2_list


def append_data(img_list, msk_list, filename_list, num_disc_table, num_labels_list, np_image, masks, overview_table, img_file_we, mask_key):
    """Append data to the respective lists"""
    img_list.append(np_image)
    msk_list.append(masks[mask_key])
    filename_list.append(img_file_we)
    curr_value=overview_table[overview_table["new_file_name"]==img_file_we]["num_discs"]
    num_disc_table.append(curr_value.iloc[0])
    labels_real=measure.label(masks[mask_key])
    num_labels_list.append(np.max(np.unique(labels_real)))


def save_model_weights(model, file_path):
    """Save model weights"""
    torch.save(model.state_dict(), file_path)


def load_model_weights(model, file_path):
    """Load model weights"""
    model.load_state_dict(torch.load(file_path))


def binary_threshold(x):
    """Apply a threshold (0.5) to a tensor"""
    return torch.where(x<0.5, torch.tensor(0), torch.tensor(1))


def plot_training_metrics(epochs, train_losses, val_losses, train_ious, train_dices, val_ious, val_dices):
    """Plot training metrics including loss, IOU, and Dice scores"""
    epochs_range=range(epochs)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_ious, label='Train IOU')
    plt.plot(epochs_range, val_ious, label='Validation IOU')
    plt.title('Training and Validation IOU')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_dices, label='Train Dice')
    plt.plot(epochs_range, val_dices, label='Validation Dice')
    plt.title('Training and Validation Dice')
    plt.legend()
    plt.show()


def inference(data_loader, model, batch_size, inference_data, test_size):
    """Compute mean values across the dataset of IOU and Dice scores"""
    total_iou, total_dice=0, 0
    inference_data_size=np.round(inference_data.shape[0]*test_size)
    output_num_list, iou_list, dice_list=[], [], []
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad(): 
        for i, (labels, inputs) in enumerate(data_loader):
            inputs=inputs.float().to(device)
            labels=labels.float().to(device)
            logits=model(inputs)
            output=binary_threshold(logits)
            output_num_list.append(output.cpu())
            for j in range(inputs.size(0)):  
                current_iou=get_iou_train(output[j, 0], labels[j, 0])
                total_iou+=current_iou
                iou_list.append(current_iou)
                current_dice=get_dice(output[j, 0], labels[j, 0])
                total_dice+=current_dice
                dice_list.append(current_dice)
            if (i+1)*batch_size>inference_data_size:
                break
    mean_iou=total_iou/len(iou_list)
    mean_dice=total_dice/len(dice_list)
    print(f"mean_iou: {mean_iou}, mean_dice: {mean_dice}")
    return output_num_list, iou_list, dice_list


def process_mismatched_data(img_t1_MRI, msk_t1_MRI, filename_t1, num_labels_real_t1_list, num_disc_table_t1, rad_grad_table):
    "Removes mismatched patient data from MRI datasets and updates relevant tables based on discrepancies"
    mismatch_t1=find_differences(num_labels_real_t1_list, num_disc_table_t1)
    img_t1_MRI_new=np.delete(img_t1_MRI, mismatch_t1, axis=0)
    msk_t1_MRI_new=np.delete(msk_t1_MRI, mismatch_t1, axis=0)
    filename_t1_new=np.delete(filename_t1, mismatch_t1)
    num_disc_table_t1_new=np.delete(num_disc_table_t1, mismatch_t1)
    num_labels_real_t1_list_new=np.delete(num_labels_real_t1_list, mismatch_t1)

    patients_to_discard=[int(filename_t1[mismatch_t1[0]][:-3]), int(filename_t1[mismatch_t1[1]][:-3])]
    rad_grad_table_fin=rad_grad_table[(rad_grad_table["Patient"]!=patients_to_discard[0]) & (rad_grad_table["Patient"]!=patients_to_discard[1])]

    return img_t1_MRI_new, msk_t1_MRI_new, filename_t1_new, num_disc_table_t1_new, num_labels_real_t1_list_new, rad_grad_table_fin


def find_differences(list1, list2):
    "Identifies and returns the indices where two lists differ, accounting for differing lengths"
    length=min(len(list1), len(list2))
    differing_indices=[i for i in range(length) if list1[i]!=list2[i]]
    if len(list1)!=len(list2):
        extra_length=max(len(list1), len(list2))
        differing_indices.extend(range(length, extra_length))
    
    return differing_indices


def generate_outputs(autoencoder, data_loader, device):
    "Generate thresholded outputs for a dataset using a trained autoencoder"
    output_num=[]
    for i, image in enumerate(data_loader):
        inputs=image[0].float().to(device)
        output=autoencoder(inputs)
        outputs=binary_threshold(output)
        output_fin=outputs.detach().cpu().numpy()
        output_num.append(output_fin[0][0])
    return output_num


def extract_latent_representations(autoencoder, data_loader, device):
    "Extract latent representations from the encoder for a given dataset"
    autoencoder.eval()
    latent_representations=[]
    for i, image in enumerate(data_loader):
        inputs=image[0].float().to(device)
        latent_representation=autoencoder.encoder(inputs)
        latent_representations.append(latent_representation.detach().cpu().numpy())
    latent_representations=np.squeeze(np.stack(latent_representations))
    return latent_representations


def create_features_dataframe(disc_height_list, ap_width_list, lat_width_list, sag_angle, trasv_angle, front_angle, latent_representation):
    """Create dataframes for geometric features, latent features, and their combination."""
    data_height_array=np.array(disc_height_list).reshape(-1,1)
    data_ap_width_array=np.array(ap_width_list).reshape(-1,1)
    data_lat_width_array=np.array(lat_width_list).reshape(-1,1)
    data_angle_sag_list_array=np.array(sag_angle).reshape(-1,1)
    data_angle_trasv_list_array=np.array(trasv_angle).reshape(-1,1)
    data_angle_front_list_array=np.array(front_angle).reshape(-1,1)

    geom_features_array=np.concatenate([
        data_height_array, 
        data_ap_width_array, 
        data_lat_width_array, 
        data_angle_sag_list_array, 
        data_angle_trasv_list_array, 
        data_angle_front_list_array
    ], axis=1)
    geom_features_df=pd.DataFrame(
        geom_features_array, 
        columns=["disc height","ap width","lat width", "sag angle", "axial angle", "front angle"]
    )

    latent_features_df=pd.DataFrame(
        np.array(latent_representation), 
        columns=[f"Feature {i}" for i in range(len(latent_representation[0]))]
    )
    
    all_features_df=pd.concat([latent_features_df, geom_features_df], axis=1)

    return geom_features_df, latent_features_df, all_features_df


def plot_correlation_matrix(features_df, cmap='PRGn', annot=True):
    """Generate and visualize the correlation matrix for the given DataFrame."""
    correlations = features_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=annot, cmap=cmap, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()


def process_metadata_and_filter(metadata_file, filenames, test_size=42):
    """Process metadata and filter it based on the subject IDs in the test filenames"""
    
    metadata=pd.read_excel(metadata_file)
    metadata["Subject ID"]=metadata["Subject ID"].replace('scoliosis', np.nan)
    metadata["Subject ID"]=pd.to_numeric(metadata["Subject ID"], errors='coerce')
    metadata["Subject ID"]=metadata["Subject ID"].ffill()
    metadata["Subject ID"]=metadata["Subject ID"].astype(int)
    filename_int=[int(item.split('_')[0]) for item in filenames]
    filename_int_test=filename_int[-test_size:]
    filtered_metadata=metadata[metadata['Subject ID'].isin(filename_int_test)]
    return filtered_metadata



def calculate_feature_range(latent_representations, feature_index):
    """Calculate the range of a given feature in the latent space"""
    latent_representations_tensors = torch.tensor(latent_representations).cpu()
    feature_min = latent_representations_tensors[:, feature_index].min()
    feature_max = latent_representations_tensors[:, feature_index].max()
    return feature_min, feature_max


def calculate_fixed_values(latent_representations):
    """Calculate the fixed values for latent features"""
    latent_representations_tensors = torch.tensor(latent_representations).cpu()
    mean_values = (
        torch.max(latent_representations_tensors, dim=0).values-
        torch.min(latent_representations_tensors, dim=0).values
    ) / 2
    return mean_values


def print_cuda_memory_stats():
    """Print CUDA memory statistics"""
    print(f"Current memory allocated (GB): {torch.cuda.memory_allocated() / (1024 ** 3):.2f}")
    print(f"Max memory allocated (GB): {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}")
    print(f"Current memory cached (GB): {torch.cuda.memory_reserved() / (1024 ** 3):.2f}")
    print(f"Max memory cached (GB): {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f}")
