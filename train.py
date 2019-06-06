#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# ## Configurations

# In[55]:


class ImageConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cocotext"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + text

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 1280

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    
config = ImageConfig()
config.display()


# ## Notebook Preferences

# In[56]:


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Dataset
# 
# Create a synthetic dataset
# 
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
# 
# * load_image()
# * load_mask()
# * image_reference()

# In[80]:


import coco_text

class ImageDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self, ann_path, dataDir, train):
        super(ImageDataset, self).__init__()
        # link annotation
        self.ct = coco_text.COCO_Text(ann_path)
        # save the path to picture data
        self.dataDir = dataDir
        self.add_class("cocotext", 1, "text")
        if train:
            db = self.ct.train
        else:
            db = self.ct.val
        imgIds = self.ct.getImgIds(imgIds=db, 
                    catIds=[('legibility','legible')])
        for imgId in imgIds:
            image_array = self.ct.loadImgs(imgId)[0]
            # '%s/%s'%(self.dataDir,img['file_name']) is to get the file path
            self.add_image("cocotext", 
                           image_id = imgId, 
                           path = self.dataDir + "/" + image_array['file_name'],
                           height = image_array["height"], 
                           width = image_array["width"])
    
    def coor_to_mask(coors, width, height):
        # coors is an array of [x1, y1, x2, y2, ...]
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(coors, outline=1, fill=1)
        mask = np.array(img)
        return mask
    
    def load_image(self, image_id):
        # here, the image id is referred to internel image id
        # instead of the id of the image in the annotated dataset
        I = super(ImageDataset, self).load_image(image_id)
        return I

    def image_reference(self, image_id):
        # return the related information of the given image
        info = self.image_info[image_id]
        if info["source"] == "cocotext":
            return info
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        # here the ground truth mask is on for all characters, legible and illegible
        # include catIds=[('legibility','legible')] in getAnnIds to have illegible removed.

        info = self.image_info[image_id]
        coco_id = info['id']
        annIds = self.ct.getAnnIds(imgIds = coco_id)
        anns = self.ct.loadAnns(annIds)
        mask = np.zeros([info['height'], info['width'], len(anns)], dtype=np.uint8)
        class_ids = np.ones([len(anns)], dtype=np.uint8)
        for i in range(len(anns)):
            mask[:, :, i] = ImageDataset.coor_to_mask(anns[i]['mask'], info["width"], info["height"])
        return mask, class_ids

# ann_path = "cocotext.v2.json"
# dataDir = "./train2014"
# dataset_train = ImageDataset(ann_path, dataDir)
# arg 1 annotation path
# arg 2 train image path
# arg 3 validation image path
dataset_train = ImageDataset(ann_path = sys.argv[1], dataDir = sys.argv[2], train = True)
dataset_val = ImageDataset(ann_path = sys.argv[1], dataDir = sys.argv[3], train = False)
dataset_train.prepare()
dataset_val.prepare()




# # Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 1)
# # image_ids = [2]
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, limit = 1)


# ## Create Model

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[8]:


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
print("weights loaded")

# ## Training
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

# In[10]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1000, 
            layers='heads')
print("finished training head layers")

# In[ ]:


# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=200, 
            layers="all")


# In[ ]:


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
model.keras_model.save_weights(model_path)
