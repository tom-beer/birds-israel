import json

import torch
from torch.nn import functional as F
from torch import nn
import torchvision.models as tvmodels
import pytorch_lightning as pl
import torchmetrics

from timm import create_model as create_timm_model
import wandb

from constants import INPUT_IMAGE_SIZE

pl.seed_everything(hash("setting random seeds") % 2**32 - 1)


class LitMLP(pl.LightningModule):

    def __init__(self, batch_size, n_classes=10, lr=1e-4):
        super().__init__()
        self.batch_size = batch_size

        self.feature_extractor, num_filters = get_feature_extractor()
        self.classifier = nn.Linear(num_filters, n_classes)

        # self.model = create_timm_model('resnet50d', pretrained=True, num_classes=n_classes)
        # self.model.eval()
        # self.model.get_classifier().train()

        # log hyperparameters
        self.save_hyperparameters()

        # compute the accuracy -- no need to roll your own!
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.img_class_map = get_img_class_map()

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        x = F.log_softmax(x, dim=1)
        return x

    def predict_app(self, x):
        self.eval()
        _, y_hat = self.forward(x).max(1)
        return {'class_id': y_hat.item(), 'class_name': self.img_class_map[str(y_hat.item())]}

    # convenient method to get the loss on a batch
    def loss(self, xs, ys):
        logits = self(xs)  # this calls self.forward
        loss = F.nll_loss(logits, ys)
        return logits, loss

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        # logging metrics we calculated by hand
        self.log('train/loss', loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(preds, ys)
        self.log('train/acc', self.train_acc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        self.test_acc(preds, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
        dummy_input = torch.zeros((self.batch_size, *(3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)), device=self.device)
        model_filename = "model_final.onnx"
        self.to_onnx(model_filename, dummy_input, export_params=True)
        wandb.save(model_filename)

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_acc(preds, ys)

        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log('valid/acc_epoch', self.valid_acc)

        return logits

    def validation_epoch_end(self, validation_step_outputs):
        dummy_input = torch.zeros((self.batch_size, *(3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)),
                                  device=self.device)
        model_filename = f"model_{str(self.global_step).zfill(5)}.onnx"
        torch.onnx.export(self, dummy_input, 'latest_run' + model_filename, opset_version=11,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}}
                          )
        wandb.save(model_filename)

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
             "global_step": self.global_step})


def get_img_class_map():
    with open('index_to_name.json') as f:
        img_class_map = json.load(f)
    return img_class_map


def get_feature_extractor():
    # backbone = tvmodels.resnet50(pretrained=True)
    backbone = create_timm_model('resnet50d', pretrained=True)
    num_filters = backbone.fc.in_features
    layers = list(backbone.children())[:-1]
    return nn.Sequential(*layers), num_filters
