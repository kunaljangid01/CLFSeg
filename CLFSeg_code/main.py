import os
import tensorflow as tf
import albumentations as albu
import numpy as np
import gc
import pickle
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score

from tensorflow.keras.losses import BinaryCrossentropy
from utils.dataloader import DataLoader
from model import CLFSeg
from medpy import metric
import keras.backend as K
from tqdm import tqdm


def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    if ground_truth.dtype != tf.float32 or predictions.dtype != tf.float32:
        ground_truth = K.cast(ground_truth, tf.float32)
        predictions = K.cast(predictions, tf.float32)

    ground_truth = K.flatten(ground_truth)
    predictions = K.flatten(predictions)
    intersection = K.sum(predictions * ground_truth)
    union = K.sum(predictions) + K.sum(ground_truth)

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice


def total_loss(self, y_true, y_pred):
    y_true = K.cast(y_true, tf.float32)
    y_pred = K.cast(y_pred, tf.float32)
    bce = BinaryCrossentropy()
    bin_loss = bce(y_true, y_pred)
    dsc_loss = dice_metric_loss(y_true, y_pred)
    return bin_loss + dsc_loss



class Trainer:
    def __init__(self, IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS, OUT_CHANNELS, FILTERS, EPOCHS, SAVEPATH, SAVENAME):
        self.model = CLFSeg.create_model(
            img_height=IMG_HEIGHT, 
            img_width=IMG_WIDTH, 
            input_chanels=IN_CHANNELS, 
            out_classes=OUT_CHANNELS, 
            starting_filters=FILTERS
        )

        learning_rate = 1e-4
        seed_value = 58800
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer, loss=total_loss)

        self.epochs = EPOCHS
        self.savepath = SAVEPATH
        self.savename = SAVENAME

    def _augment_images(self, images, masks):
        img_out, msk_out = [], []
        augmentations = albu.Compose([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ColorJitter(brightness=(0.6,1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
            albu.Affine(scale=(0.5,1.5), translate_percent=(-0.125,0.125), rotate=(-180,180), shear=(-22.5,22), always_apply=True),
        ])

        for img, msk in zip(images, masks):
            obj = augmentations(image=img, mask=msk)
            img_out.append(obj["image"]); msk_out.append(obj["mask"]);

        return np.array(img_out), np.array(msk_out)

    def train(self, X_train, X_val, y_train, y_val):
        min_loss = 2

        for epoch in tqdm(range(self.epochs)):
            img_aug, msk_aug = self._augment_images(X_train, y_train)
            self.model.fit(
                x=img_aug,
                y=msk_aug,
                epochs=1,
                batch_size=4
            )

            pred_train = self.model.predict(X_train)
            loss_train = total_loss(pred_train, y_train)

            pred_val = self.model.predict(X_val)
            loss_val = total_loss(pred_val, y_val)

            if min_loss > loss_val:
                min_loss = loss_val
                self.model.save(f"{self.savepath}/{self.savename}")


class Tester:
    def __init__(self, MODELPATH, dname, n_classes):
        self.model = tf.keras.models.load_model(
            MODELPATH, 
            custom_objects={'dice_metric_loss':dice_metric_loss, 'total_loss':total_loss}
        )

        self.savepath = MODELPATH
        self.dname = dname
        self.n_classes = n_classes
    
    def _format_multiclass(self, y_true, y_pred):
        y_pred = tf.cast(y_pred > 0.5, tf.int32)
        y_true = tf.cast(y_true > 0.5, tf.int32)
        
        class_ids = np.arange(self.n_classes)  
        y_true = y_true * class_ids
        y_pred = y_pred * class_ids                

        return y_true, y_pred
    
    def _compute_metrics(self, pred, gt):
        pred = pred.astype(int)
        gt = gt.astype(int)

        if pred.sum() > 0 and gt.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            return [dice, hd95]
        elif pred.sum() > 0 and gt.sum()==0:
            return [0, 0]
        else:
            return [0, 0]


    def test(self, images, y_true):
        y_pred = self.model(images)

        if self.dname in ["acdc"]:
            y_true, y_pred = self._format_multiclass(
                y_true, y_pred
            )

        y_true = np.ndarray.flatten(y_true.astype(bool))
        y_pred = np.ndarray.flatten(y_pred.numpy() > 0.5)

        # TODO: UNCOMMENT FOR DSC AND HD95
        # print(self._compute_metrics(y_pred, y_true))

        return jaccard_score(y_true, y_pred)
            
        

if __name__ == "__main__":

    IMG_HEIGHT = 352
    IMG_WIDTH = 352
    IN_CHANNELS = 3
    OUT_CHANNELS = 1
    FILTERS = 17
    EPOCHS = 600
    SAVEPATH = "chkpt/"
    DATANAME = "cvc-colondb"
    DATAPATH = f"data/{DATANAME}"
    SAVENAME = f"{DATANAME}/17-Filters CLFSeg"

    
    X_train, y_train = DataLoader(
        path=DATAPATH, 
        IMG_HEIGHT=IMG_HEIGHT, 
        IMG_WIDTH=IMG_WIDTH, 
        dname=DATANAME, 
        dtype="train"
    ).getdata()

    X_test, y_test = DataLoader(
        path=DATAPATH, 
        IMG_HEIGHT=IMG_HEIGHT, 
        IMG_WIDTH=IMG_WIDTH, 
        dname=DATANAME, 
        dtype="test"
    ).getdata()

    X_val, y_val = DataLoader(
        path=DATAPATH, 
        IMG_HEIGHT=IMG_HEIGHT, 
        IMG_WIDTH=IMG_WIDTH, 
        dname=DATANAME, 
        dtype="validation"
    ).getdata()


    # TODO: UNCOMMENT FOR TRAINING THE MODEL
    # trainer = Trainer(
    #     IMG_HEIGHT, 
    #     IMG_WIDTH, 
    #     IN_CHANNELS, 
    #     OUT_CHANNELS, 
    #     FILTERS, 
    #     EPOCHS, 
    #     SAVEPATH,
    #     SAVENAME
    # )

    # trainer.train(X_train, X_val, y_train, y_val)


    # TODO: UNCOMMENT FOR TESTING THE MODEL
    tester = Tester(
        MODELPATH=f"{SAVEPATH}/{SAVENAME}",
        dname=DATANAME,
        n_classes=1
    )

    print(tester.test(X_test, y_test))
    
