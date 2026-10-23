import os
import torch

class Config:
    uNetModel = "Dual"              # "Single-MSI", "Single-SAR", "Single-Both", "Dual"
    fusion_name = "Feature-scSE"    # "Feature-scSE", "Late-scSE"
    source_tag = "S2_S1"
    source_parts = ("S2", "S1")

    num_classes = 1
    image_size = 256
    num_channels_S1 = 3
    num_channels_S2 = 4

    batch_size = 16
    num_epochs = 100
    lr = 1e-3
    weight_decay = 1e-5

    # Dataset paths
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data_split"))
    data_dir = main_dir
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    
    bce_weight = 1.0
    tversky_weight = 1.0
    sobel_weight = 1.0
    
    tversky_alpha = 0.3
    tversky_beta = 0.7
    loss_reduction = "mean"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
