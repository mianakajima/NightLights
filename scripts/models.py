import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size = 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 5)
        self.conv4 = nn.Conv2d(32, 64, kernel_size = 5)
        self.conv5 = nn.Conv2d(64, 128, kernel_size = 5)
        self.fc1 = nn.Linear(128 * 9 * 9, 120) # channels * dims of last conv NN
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Lightning Model
class LitModel(pl.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def training_step(self, batch, batch_idx):

        images, labels = batch
        outputs = self.encoder(images)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        return [optimizer]

    def validation_step(self, batch, batch_idx):

        images, labels = batch
        outputs = self.encoder(images)
        val_loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", val_loss)


class InputMonitor(pl.Callback):
    # from very helpful blog on debugging Pytorch https://www.pytorchlightning.ai/blog/3-simple-tricks-that-will-change-the-way-you-debug-pytorch
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):

        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            logger = trainer.logger
            logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            logger.experiment.add_histogram("target", y, global_step=trainer.global_step)
