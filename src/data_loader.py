import torch
import numpy as np
from monai.networks.nets import SwinUNETR
from .metrics import combined_dice_bce_loss


def split_dataset(data, labels, test_size=0.25):
    """Split dataset into training and testing sets based on the test size ratio"""
    split_idx=int(len(data)*(1-test_size))
    train_data=data[:split_idx]
    test_data=data[split_idx:]
    train_labels=labels[:split_idx]
    test_labels=labels[split_idx:]
    return train_data, test_data, train_labels, test_labels


def create_dataloader(data, labels, batch_size=1, shuffle=False):
    """Create a DataLoader for the given dataset and labels"""
    dataset=torch.utils.data.TensorDataset(
        torch.tensor(labels).unsqueeze(1), torch.tensor(data).unsqueeze(1)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def initialize_model(img_size=(96,192,128), in_channels=1, out_channels=1, feature_size=48, lr=0.001):
    """Initialize the SwinUNETR model, optimizer, and loss function"""
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=True,
    ).to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn=combined_dice_bce_loss
    return model, optimizer, loss_fn


def setup_training_pipeline(
    data, labels, img_size=(96,192,128), in_channels=1, out_channels=1, feature_size=48,
    batch_size=1, lr=0.001, test_size=0.25, shuffle=True
):
    """Set up the training pipeline including data split, DataLoader, and model initialization"""
    train_data, test_data, train_labels, test_labels=split_dataset(data, labels, test_size)
    training_loader=create_dataloader(train_data, train_labels, batch_size=batch_size, shuffle=shuffle)
    testing_loader=create_dataloader(test_data, test_labels, batch_size=batch_size, shuffle=False)
    model, optimizer, loss_fn=initialize_model(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        lr=lr
    )
    return training_loader, testing_loader, model, optimizer, loss_fn


def extract_test_set(msk_t1_MRI_new, img_t1_MRI, test_size):
    """Extract the test set portion of the mask dataset based on the test size ratio"""
    len_test_set=int(img_t1_MRI.shape[0]*test_size)
    msk_t1_MRI_new_test=msk_t1_MRI_new[-len_test_set:]
    return msk_t1_MRI_new_test


def create_dataloader_from_masks(msk_t1_MRI_single_def, batch_size):
    """Create a DataLoader from processed masks with the specified batch size"""
    train_autoenc_msk_fin=torch.tensor(msk_t1_MRI_single_def).unsqueeze(1)
    dataset_img_msk_real=torch.utils.data.TensorDataset(train_autoenc_msk_fin)
    training_img_msk_real=torch.utils.data.DataLoader(dataset_img_msk_real, batch_size=batch_size, shuffle=False)
    return training_img_msk_real
