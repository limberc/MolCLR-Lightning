from pytorch_lightning.cli import LightningCLI

from mol_data import MolDataModule
from .molclr import MolCLRModule

if __name__ == '__main__':
    cli = LightningCLI(MolCLRModule, MolDataModule, seed_everything_default=42,
                       trainer_defaults={
                           'devices': 2,
                           'accelerator': 'auto',
                           'strategy': 'ddp',
                           'accumulate_grad_batches': 3,
                           'precision': '16-true',
                       })
