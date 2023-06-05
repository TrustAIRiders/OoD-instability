#!/usr/bin/env python

import os
import torch
import torchvision

from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import Trainer, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.resnet import ResNet, Bottleneck
from models.mobilenetv2 import MobileNetV2

DEVICES = [0]
MODEL_TYPE = "ResNet" # ResNet or MobileNet
DATA_TYPE = "CIFAR-10" # CIFAR-10 or CIFAR-100
BATCH_SIZE = 128
MAX_EPOCHS = -1

class DataModule(LightningDataModule):
    def __init__(self, DATA_TYPE, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        if DATA_TYPE == "CIFAR-10":
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2023, 0.1994, 0.2010)
            
        if DATA_TYPE == "CIFAR-100":
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)          

    def train_dataloader(self):
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        
        if DATA_TYPE == "CIFAR-10":
            train_set = torchvision.datasets.CIFAR10(root="./", train=True, download=True, transform=train_transforms)
            
        if DATA_TYPE == "CIFAR-100":
            train_set = torchvision.datasets.CIFAR100(root="./", train=True, download=True, transform=train_transforms)
            
        return torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        
        if DATA_TYPE == "CIFAR-10":
            test_set = torchvision.datasets.CIFAR10(root="./", train=False, download=True, transform=test_transforms)
            
        if DATA_TYPE == "CIFAR-100":
            test_set = torchvision.datasets.CIFAR100(root="./", train=False, download=True, transform=test_transforms)
        
        return torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)  

class Model(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_module = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=1e-05)
        lr_schedulers = {"scheduler": scheduler, "monitor": "train_loss"}
        return [optimizer], [lr_schedulers]         
    
    def calculate_acc_loss(self, batch):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
   
        return acc, loss
    
    def log_acc_loss(self, acc, loss, prefix):
        self.log("{}_acc".format(prefix), acc, on_step=False, on_epoch=True, logger=True)
        self.log("{}_loss".format(prefix), loss, on_step=False, on_epoch=True, logger=True)       
    
    def training_step(self, batch, batch_idx):
        acc, loss = self.calculate_acc_loss(batch)
        self.log_acc_loss(acc, loss, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        acc, loss = self.calculate_acc_loss(batch)
        self.log_acc_loss(acc, loss, "val")

for seed in range(10):
    seed_everything(seed, workers=True)
    
    if MODEL_TYPE == "ResNet" and DATA_TYPE == "CIFAR-10":
        model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=10)
        data = DataModule("CIFAR-10")
        name = "cifar10_ResNet_seed_{}".format(seed) 
    
    if MODEL_TYPE == "MobileNet" and DATA_TYPE == "CIFAR-100":
        model = MobileNetV2(num_classes=100)
        data = DataModule("CIFAR-100")
        name = "cifar100_MobileNet_seed_{}".format(seed)         

    logger = TensorBoardLogger(save_dir="./logs/", name=name, default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor="train_loss", mode="min", patience=10)

    model = Model(model)
    
    trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", devices=DEVICES, logger=logger, callbacks=[lr_monitor, early_stopping], deterministic=True)
    trainer.fit(model, datamodule=data)
    trainer.save_checkpoint("./saved_models/{}.ckpt".format(name))
