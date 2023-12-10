import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import GCN, GINet
from utils.nt_xent import NTXentLoss


class MolCLRModule(LightningModule):
    def __init__(self, model_type: str = 'gcn',
                 batch_size=512,
                 # training args
                 warm_up: int = 10, epochs: int = 100, init_lr: float = 5e-4, weight_decay: float = 1e-5,
                 # loss args
                 temperature=0.1, use_cosine_similarity=True):
        super().__init__()
        self.save_hyperparameters()
        self.warm_up = warm_up
        self.epochs = epochs
        self.init_lr = init_lr
        self.weight_decay = weight_decay

        self.loss = NTXentLoss(self.device, batch_size, temperature=temperature,
                               use_cosine_similarity=use_cosine_similarity)
        if model_type == 'gin':
            self.model = GINet()
        elif model_type == 'gcn':
            self.model = GCN()
        else:
            raise ValueError('Undefined GNN model.')

    def forward(self, x_i, x_j):
        ris, zis = self.model(x_i)
        rjs, zjs = self.model(x_j)
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        return self.loss(zis, zjs)

    def training_step(self, batch, batch_idx):
        x_i, x_j = batch
        loss = self.forward(x_i, x_j)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), self.init_lr,
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.epochs - self.warm_up,
            eta_min=0, last_epoch=-1
        )
        return [optimizer], [scheduler]
