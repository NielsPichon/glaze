import argparse
import datetime
import os
import pathlib
from typing import Dict, Tuple

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch.utils import tensorboard
import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification


class ImageDataset(data.Dataset):
    """Loader for the glaze image dataset."""
    def __init__(self, data_dir: pathlib.Path, n_atm: int):
        self._files = list(data_dir.glob('*.npy'))
        self._n_atm = n_atm

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        datum = np.load(self._files[idx], allow_pickle=True).item()
        datum = {k: torch.tensor(v) for k, v in datum.items()}
        datum['cone_mask'] = torch.tensor(
            1.0 if 'cone' in datum else 0.0, dtype=torch.float32)
        datum['atmosphere_mask'] = torch.tensor(
            1.0 if 'atmosphere' in datum else 0.0, dtype=torch.float32)
        # the cone is scaled down to be in range 0-1, 13 being the max cone seen
        # in the source data.
        datum['cone'] = datum.get(
            'cone', torch.tensor(0.0)) / 13
        datum['atmosphere'] = datum.get(
            'atmosphere', torch.zeros(self._n_atm))

        for k in ['amount', 'extras', 'cone', 'atmosphere']:
            datum[k] = datum[k].to(torch.float32)

        for k in datum:
            if datum[k].dim() == 0:
                datum[k] = datum[k].unsqueeze(0)

        return datum


class GlazeModel(torch.nn.Module):
    def __init__(self, n_atm: int, n_ing: int):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224")
        self.backbone = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224").vit
        self.atm_fc = nn.LazyLinear(n_atm)
        self.ingredients_fc = nn.LazyLinear(n_ing)
        self.extras_fc = nn.LazyLinear(n_ing)
        self.cone_fc = nn.LazyLinear(1)
        self.device = torch.device('cpu')

    def get_backbone_params(self):
        """Returns the parameters fgrom the backbone model only."""
        return self.backbone.parameters()

    def get_head_params(self):
        """Returns all the parameters form the heads."""
        return (*self.atm_fc.parameters(),
                *self.ingredients_fc.parameters(),
                *self.extras_fc.parameters(),
                *self.cone_fc.parameters())

    def to(self, *args, **kwargs):
        """Overriding to() to keep self.device in sync with the model's device.
        """
        if isinstance(args[0], torch.device):
            self.device = args[0]
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(x, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # ViTForImageClassification only uses the first token (CLS) for
        # classification as it should hold all the information on the image
        out = self.backbone(**inputs)[0][:, 0, :]
        atm = self.atm_fc(out)

        ingredients = self.ingredients_fc(out)
        extras = self.extras_fc(out)
        # we normalize because the sum should be 1.
        # A softmax would skew the distribution towards having a single
        # ingredient which is not what we want. Sigmoid would also skew the
        # distribution towards having either a lot or none of any given
        # ingredient. Because we want to have a positive value for each but
        # not necessarily a set maximum, we also apply a ReLu before the
        # normalization.
        ingredients = F.relu(ingredients)
        ingredients = ingredients / (ingredients.sum(dim=1, keepdim=True)
                                     + torch.finfo(ingredients.dtype).eps)
        cone = self.cone_fc(out)
        return ingredients, extras, atm, cone

def get_shapes(src: pathlib.Path) -> Tuple[int, int]:
    """Get the shapes of the dataset.

    Args:
        src (pathlib.Path): The path to the dataset.

    Returns:
        int: The number of possible ingredients.
        int: The number of possible atmospheres.
    """
    with open(src / 'ingredients.txt', 'r') as f:
        n_ing = len(f.readlines())

    with open(src / 'atmosphere.txt', 'r') as f:
        n_atm = len(f.readlines())

    return n_ing, n_atm


def step(device: torch.device,
         model: nn.Module,
         batch: Dict[str, torch.Tensor]):
    ingredients, extras, atm, cone = model(batch['input'])

    ing_loss = torch.nn.functional.mse_loss(ingredients,
                                            batch['amount'].to(device))

    extra_loss = torch.nn.functional.mse_loss(extras,
                                             batch['extras'].to(device))

    batch['cone_mask'] = batch['cone_mask'].to(device)
    cone_count = batch['cone_mask'].sum() + 1e-6
    # no activation on cone because we want an actual value.
    # The target value is scaled down to be between 0 and 1.
    cone_loss = (
        torch.nn.functional.mse_loss(
            cone, batch['cone'].to(device), reduction='none')
        * batch['cone_mask']
    ).sum() / cone_count

    batch['atmosphere_mask'] = batch['atmosphere_mask'].to(device)
    atm_count = batch['atmosphere_mask'].sum() + 1e-6
    # the atm is a multi-class classification problem so we use a masked
    # sigmoid cross entropy
    atm_loss = (
        torch.nn.functional.cross_entropy(
            F.sigmoid(atm), batch['atmosphere'].to(device), reduction='none')
        * batch['atmosphere_mask']
    ).sum() / atm_count

    loss = ing_loss + extra_loss + cone_loss + atm_loss
    return {
        'loss': loss,
        'ingredients': ing_loss,
        'extras': extra_loss,
        'cone': cone_loss,
        'atmosphere': atm_loss
    }


def prepare_dataloader(root_dir: pathlib.Path,
                       batch_size: int,
                       use_cuda: bool,
                       n_atm: int,
                       split: str = 'train') -> data.DataLoader:
    """Prepares the dataloader for a dataset split.

    Args:
        root_dir (pathlib.Path): root data directory containing the data
            split directory
        batch_size (int): Batch size
        use_cuda (bool): Whether cuda will be used.
        n_atm (int): Number of possible atmospheres.
        split (str): Data split. Defaults to 'train'.

    Returns:
        data.DataLoader: DataLoader for the required data split.
    """
    dataset = ImageDataset(root_dir / split, n_atm)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=min(8, os.cpu_count() - 1),
                                 pin_memory=use_cuda,
                                 drop_last=True,
                                 prefetch_factor=2,
                                 persistent_workers=True
                                )
    return dataloader


def run_training(src: pathlib.Path,
                 ckpt_path: pathlib.Path,
                 batch_size: int = 32,
                 epochs: int = 10,
                 lr: float = 1e-3,
                 train_head_only: bool = True):
    """Run the training loop."""
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
    logger.info(f'Using device {device}')

    ckpt_path = ckpt_path / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    ckpt_path.mkdir(parents=True, exist_ok=True)

    n_ing, n_atm = get_shapes(src)

    logger.info('initializing dataloaders')
    train_loader = prepare_dataloader(src, batch_size, use_cuda, n_atm, 'train')
    eval_loader = prepare_dataloader(src, batch_size, use_cuda, n_atm, 'eval')

    logger.info('initializing model')
    model = GlazeModel(n_atm, n_ing).to(device)
    optimizer = torch.optim.Adam(lr=lr, params=model.get_head_params()
                                 if train_head_only else model.parameters())

    if train_head_only:
        logger.info('Freezing backbone')
        model.backbone.requires_grad_(False)
        model.backbone.eval()

    logger.info('initializing tensorboard')
    writer = tensorboard.SummaryWriter(ckpt_path)

    logger.info('Training model')
    epoch_bar = tqdm.tqdm(range(epochs))
    for epoch in epoch_bar:
        epoch_bar.set_description(f'Epoch {epoch} / {epochs}')
        progress = tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                             leave=False)
        for i, batch in progress:
            with torch.autocast(device_type="cuda" if use_cuda else "cpu"):
                loss = step(device, model, batch)
            loss['loss'].backward()
            progress.set_description(f'[TRAIN] loss: {loss["loss"].item():.4f}')
            optimizer.step()
            optimizer.zero_grad()

            for k, v in loss.items():
                writer.add_scalar(f'Loss/{k}/train', v.item(),
                                  epoch * len(train_loader) + i)
        progress = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader),
                             leave=False)
        for i, batch in progress:
            with torch.inference_mode():
                loss = step(device, model, batch)
            progress.set_description(f'[EVAL] loss: {loss["loss"].item():.4f}')
            for k, v in loss.items():
                writer.add_scalar(f'Loss/{k}/eval', v.item(),
                                  epoch * len(eval_loader) + i)

        torch.save(model.state_dict(), ckpt_path / f'epoch_{epoch}.pt')
        torch.save(optimizer.state_dict(), ckpt_path / f'epoch_{epoch}_opt.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=pathlib.Path, default='data/',
                        help='Where the data is stored')
    parser.add_argument('--out', type=pathlib.Path, default='checkpoints/',
                        help='Where to save the model')
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train-all', dest='train_head_only',
                        action='store_false',
                        help='Train the whole model rather than only '
                        'the heads.')
    args = parser.parse_args()

    run_training(src=args.src,
                 ckpt_path=args.out,
                 batch_size=args.batchsize,
                 epochs=args.epochs,
                 lr=args.lr,
                 train_head_only=args.train_head_only)
