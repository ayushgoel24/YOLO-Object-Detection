import pytorch_lightning as pl
import torch
import torch.nn as nn

from .conv_layer import ConvLayer
from .loss import YOLOLoss
from .transposed_conv_layer import TransposedConvLayer

class YOLO(pl.LightningModule):
    """
    YOLO (You Only Look Once) Deep Learning model for object detection.
    """
    def __init__(self):
        super(YOLO, self).__init__()

        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            ConvLayer(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
            ConvLayer(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            ConvLayer(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            ConvLayer(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            ConvLayer(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            ConvLayer(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)
        )

        # Decoder layers
        self.decoder_layers = nn.Sequential(
            TransposedConvLayer(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1),
            TransposedConvLayer(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Lists to store training and validation losses and predictions
        self.train_losses = []
        self.val_losses = []
        self.predictions_per_batch = []
        self.predictions_per_epoch = []

    def forward(self, x):
        x = self.feature_layers(x)
        return self.decoder_layers(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.view(len(images), 3, 128, 128).float()
        labels = labels.float()

        outputs = self(images)
        self.predictions_per_batch.append(outputs)

        loss = YOLOLoss.compute_loss(outputs, labels)
        self.train_losses.append(loss.item())
        self.log('training_loss', loss)
        return loss

    def on_train_epoch_end(self):
        self.predictions_per_epoch.append(self.predictions_per_batch)
        self.predictions_per_batch.clear()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.view(len(images), 3, 128, 128).float()
        labels = labels.float()

        outputs = self(images)
        loss = YOLOLoss.compute_loss(outputs, labels)
        self.val_losses.append(loss.item())
        self.log('validation_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)