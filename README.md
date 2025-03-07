# Lane Detection in CARLA Using BiSeNetV2
This is a lane detection program implemented in CARLA Simulator. This program detects lanes and segments each of the lanes. I utilized BiSeNetV2 (Bilateral Segmentation Network V2) developed by Changqian Yu et al as explained in their paper: https://arxiv.org/abs/2004.02147 for the model architecture. For the data type itself, the model inputs a RGB image with the size of (256,512,3) and outputs a binary image consisting of lane or non lane pixels and a instance segmentation image that consist of area segmentation for each lane. Both of the output images are multiply by each other to get the final image of detected lanes as illustrated on the picture below.

https://user-images.githubusercontent.com/91169005/179198553-b5b0b2fb-d0b3-4d9a-bf48-e0a269b445c6.mp4


# Model
The model BiSeNetV2 that I used is shown on the figure below.
![image](https://user-images.githubusercontent.com/91169005/175543627-572fa5ca-05ad-4ef1-b815-f82a599dc19b.png)
![image](https://user-images.githubusercontent.com/91169005/175543750-dfc395a0-c008-4efa-9564-7e7f0392026d.png)

# Dataset
The dataset that I used was images extracted from CARLA Simulator with Lane Detection Dataset Extractor that is developed by Github user, Glutamat42 (https://github.com/Glutamat42/Carla-Lane-Detection-Dataset-Generation). It extracts images needed for lane detection training: (1) RGB image for the input, (2) lane binary segmentation image and (3) instance segmentation image for the ground truth. Because it only allows to set the weather setting to be Clear Noon, I modified the code to be able to custom the weather setting. The dataset I used can be found on my google drive: ...().
# Configuration
You will be able to find 'lanenet.yml' consisting the configuration of how the dataset, training, and inference setting would do. The resize image is set to be [256, 512]

# Installation
```
conda create -n lanenet python==3.9
conda activate lanenet
pip install -r requirements.txt
```

# Training Output 
The training output and records will be saved in the following structure:
```
/train_out
└── (date and time)
    ├── weight_epoch.weights.h5
    ├── final.weights.h5
    ├── history.npy
    ├── plot_fig
    │   ├── training and validation loss.png
    │   ├── (other plots and metrics visualization)
    └── (additional logs and records)
```