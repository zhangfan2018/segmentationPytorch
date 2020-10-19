# segmentationPytorch: The unified segmentation algorithm framework for segmentation of tissue, organ and lesion in three-dimension format.

## History
* \[2020/09/22] Init this project.

## Introduction
This is the implementation code of segmentation algorithm framework in torch. The detail structure is as follow.  
* data_processor: Data processor, including data loader, augmentation, re-sample etc.
* utils: utils tools , including image and mask processing tools, logging tools and files processing tools.
* network: UNet blocks and a variety of different UNet variants.
* model_utils: the utils of analysis model complexity, prune and compression model etc.
* runner: the procedure of training and test model, including configuration of hyperparameters, definition of loss function and evaluation metrics.
* experiments: new experiments can be placed in current file.

## Environment
* The code is developed using python 3.6 on Ubuntu 16.04. The code is developed and tested using 4 NVIDIA 2080Ti GPU cards. 
* Docker container(optional), namely docker-docker-compose-python.yml.
* You also need to install apex, torch-1.5.1, tensorboardX and prefetch_generator.

## Quick start
### Data prepare
The procedures of data preprocess as flows.
1. Convert image and mask format from dicom to nii.gz.
2. Analysis the bounding box of mask.
3. Crop and re-sample image and mask.
4. Convert the folder of image data into csv file.
5. Divide dataset into training and validation dataset.
The final data looks like this:
````
${SEG_ROOT}
|-- dataset
 -- ori_data
    -- images
    -- masks
 -- processed_data
    -- crop_256_192_256
       -- images
       -- masks
 -- csv
    -- train.csv
    -- val.csv
````

### Training and Test
* The shell of distribution training
````
nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port 29503 train.py &
````

## Experiments
### segmentation of rib and center-line
1. The data address is as flows.
````
* image_dir: "/fileser/CT_RIB/data/image_refine/"
* mask_dir: "/fileser/CT_RIB/data/mask_refine/"
* csv_dir: "/fileser/zhangfan/DataSet/lung_rib_data/csv/"
````
2. The code folder look like this:
```
${SEG_ROOT}
|-- experiments
 -- rib_centerline_seg
    -- data_process
    -- rib_coarse_seg
    -- rib_centerline_fine_seg
    -- rib_label24_seg
```
3. Results

| Experiment         | Backbone | Input size  | #Params | GFLOPs |  Dice | label acc.|
|--------------------|----------|-------------|---------|--------|-------|-----------| 
| Rib coarse seg     | UNet     | 128*128*128 |   -     |   -    |   -   |     -     |
| Rib fine seg       | UNet     | 224*160*224 |   -     |   -    |   -   |     -     | 
