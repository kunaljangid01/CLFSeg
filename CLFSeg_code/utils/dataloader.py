import glob
import nibabel as nib
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from skimage.io import imread
from tqdm import tqdm

import glob
import os
import random
import h5py
import numpy as np
from tqdm import tqdm


class DataLoader:
    def __init__(self, path, IMG_HEIGHT, IMG_WIDTH, dname, dtype):
        self.path = path
        self.IMAGE_PATH = f"{path}/{dtype}/images/"
        self.MASK_PATH = f"{path}/{dtype}/masks/"
        print(self.IMAGE_PATH, self.MASK_PATH)
        self.H, self.W = IMG_HEIGHT, IMG_WIDTH
        self.dname = dname
        self.dtype = dtype #train/test/val
        
        if dname in ["cvc-colondb", "cvc-clinicdb", "kvasir", "etis-laribpolypdb"]:
            self.total = glob.glob(self.IMAGE_PATH + "*.jpg")
        elif dname in ["acdc"]:
            with open(f"{path}/train.list", "r") as file:
                self.total = [fname.rstrip("\n") for fname in file.readlines()]
    
    def _preprocess_uniclass(self):
        X = np.zeros((len(self.total), self.H, self.W, 3), dtype=np.float32)
        Y = np.zeros((len(self.total), self.H, self.W), dtype=np.uint8)

        for idx, fpath in tqdm(enumerate(self.total)):
            img_path = fpath
            msk_path = fpath.replace("images", "masks")
            
            img = imread(img_path)
            msk = imread(msk_path)

            # Image
            pil_img = Image.fromarray(img).resize((self.H, self.W))
            img = np.array(pil_img)
            X[idx] = img / 255

            # Mask
            pil_msk = Image.fromarray(msk).resize((self.H, self.W), resample=Image.LANCZOS)
            msk = np.array(pil_msk)
            msk = np.where(msk >= 127, 1, 0)
            Y[idx] = msk

        return X, np.expand_dims(Y, axis=-1)


    def _preprocess_multiclass(self):
        images, labels = [], []

        for case in tqdm(self.total):
            if self.dtype == "train":
                file = h5py.File(f"{self.path}/data/slices/{case}.h5", "r")
            else:
                file = h5py.File(f"{self.path}/data/{case}.h5", "r")

            img = file["image"][:]
            msk = file["label"][:]

            if len(img.shape) == 2:
                img = img[np.newaxis, ...]
                msk = msk[np.newaxis, ...]
            
            assert img.shape[0] == msk.shape[0]
            for idx in range(img.shape[0]):
                img, msk = img[idx, ...], msk[idx, ...]
                
                pil_img = Image.fromarray(img).resize((self.H, self.W))
                img = np.array(pil_img).astype(np.float32)

                pil_msk = Image.fromarray(msk).resize((self.H, self.W), resample=Image.LANCZOS)
                msk = np.array(pil_msk).astype(np.uint8)
            
                lab = np.zeros((msk.shape[0], msk.shape[1], 4), dtype=msk.dtype)
                lab[..., 0] = (msk == 0)
                lab[..., 1] = (msk == 1)
                lab[..., 2] = (msk == 2)
                lab[..., 3] = (msk == 3)
                
                images.append(img)
                labels.append(lab)
                

        images = np.array(images)[..., np.newaxis]
        labels = np.array(labels)

        return images, labels

    def getdata(self,):
        if self.dname in ["cvc-colondb", "cvc-clinicdb", "kvasir", "etis-laribpolypdb"]:
            return self._preprocess_uniclass()
        elif self.dname in ["acdc"]:
            return self._preprocess_multiclass()
        else:
            raise Exception("dname not found!")
