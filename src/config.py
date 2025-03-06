import torch

config={
    #Paths
    "curr_path": "/path/to/dataset",
    "save_path": "/path/to/save/dataset.pth",
    "path_weights": "/path/to/weights.pth",

    #Image and Mask Settings
    "target_resolution": (1.4, 1.2, 1.5),
    "img_size": (96, 192, 128),

    #Model Parameters - segmentation
    "in_channels": 1,
    "out_channels": 1,
    "feature_size": 48,

    #Training Parameters - segmentation
    "batch_size": 4,
    "lr": 0.001,
    "num_epochs": 60,
    "test_size": 0.25,


    #Training Parameters - feature extraction
    "batch_size_ae": 4,
    "learning_rate_ae": 0.001,
    "num_epochs_ae": 1000,
    "load_weights_ae": True,  
    "weights_path_ae":"model_weights_ae.pth",

    #Model Parameters - classification
    "f1_score_type":"micro",
    "random_state":0,

    #Device Settings
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}
