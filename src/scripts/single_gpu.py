import os
import torch
import numpy as np
import torch.multiprocessing as mp

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.evaluation.metrics import compute_metrics, save_confusion_matrix, save_metrics
from src.data.data_loader import DigiFace
from src.data.preprocess import split_data
from src.losses.cosface import CosFaceLoss
from src.models.partfVit import PartFVitWithLandmark
from src.models.concat import ConcatModelWithLoss


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        device,
        save_every,
        checkpoint_path,
        experiment_dir,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = device
        self.save_every = save_every
        self.checkpoint_path = checkpoint_path
        self.experiment_dir = experiment_dir

        self.train_metrics = {"losses": []}
        self.val_metrics = {"losses": []}

        self.cache = []
        self.model.to(self.device)

    def _run_batch(self, source, target, is_train=True):
        if is_train:
            self.model.train()
            self.optimizer.zero_grad()

        else:
            self.model.eval()

        y_score = self.model(source, target)
        loss = self.criterion(y_score, target)

        if is_train:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.train_metrics = compute_metrics(self.train_metrics, target, y_score)
            self.train_metrics["losses"].append(loss.item())

        else:
            self.val_metrics = compute_metrics(self.val_metrics, target, y_score)
            self.val_metrics["losses"].append(loss.item())
            self.cache.append(
                (
                    y_score.detach().cpu().numpy(),
                    target.detach().cpu().numpy().astype(np.int64),
                )
            )

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_loader))[0])
        print(
            f"[GPU {self.device}] | Epoch: {epoch+1} | batchsize: {b_sz} | steps: {len(self.train_loader)}"
        )
        for i, (source, targets) in enumerate(self.train_loader):
            source, targets = source.to(self.device), targets.to(self.device)
            self._run_batch(source=source, target=targets, is_train=True)

    def _run_eval(self, epoch):
        self.cache = []
        for i, (source, targets) in enumerate(self.val_loader):
            source, targets = source.to(self.device), targets.to(self.device)
            self._run_batch(source=source, target=targets, is_train=False)

        save_confusion_matrix(self.cache, self.experiment_dir, epoch)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        torch.save(ckp, self.checkpoint_path)
        print(f"epoch {epoch+1}: Training checkpoint saved at {self.checkpoint_path}")

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self._run_epoch(epoch)
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch)

            self._run_eval(epoch)
        if num_epochs % self.save_every != 0:
            self._save_checkpoint(num_epochs)


def load_train_obj(config):
    train_path, val_path, test_path = split_data(
        input_path=config["data_path"],
        output_path="./data/",
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        verbose=True,
        num_identities=config["num_identities"],
    )

    train_data = DigiFace(path=train_path)
    val_data = DigiFace(path=val_path)
    test_data = DigiFace(path=test_path)

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=True)

    part_fvit = PartFVitWithLandmark(
        num_identites=train_data.num_identities,
        num_landmarks=config["num_landmarks"],
        in_channels=config["num_channels"],
        image_size=config["image_width"],
        feat_dim=config["feat_dim"],
        mlp_dim=config["mlp_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )

    cls_pred = CosFaceLoss(
        num_classes=config["num_identities"],
        feat_dim=config["feat_dim"],
        margin=config["margin"],
    )

    model = ConcatModelWithLoss(main_model=part_fvit, criterion=cls_pred)

    parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n.split(".")[1] == "landmark_CNN"
            ],
            "lr": config["lr"],
            "weight_decay": config["weight_decay_resnet"],
        }
    ]

    parameters += [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n.split(".")[1] != "landmark_CNN"
            ],
            "lr": config["lr"],
            "weight_decay": config["weight_decay_fViT"],
        }
    ]

    criterion = CrossEntropyLoss()

    optimizer = AdamW(parameters)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=config["warmup_epochs"], T_mult=1
    )

    kargs = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "val_loader": val_loader,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

    return kargs


def main(config, experiment_dir):
    """
    Parameters
    ----------
    config : dict
            Configuration dictionary containing all the parameters for training
    experiment_dir : str
            Path to the experiment directory where model results, states and result will be saved
    """

    device = config["device"]

    kargs = load_train_obj(config=config)

    trainer = Trainer(
        **kargs,
        save_every=config["save_every"],
        device=device,
        checkpoint_path=config["save_path"],
        experiment_dir=experiment_dir,
    )

    trainer.train(num_epochs=config["num_epochs"])

    save_metrics(
        trainer.train_metrics,
        title="Training Metrics",
        filename=os.path.join(experiment_dir, "train_metrics.jpeg"),
    )

    save_metrics(
        trainer.val_metrics,
        title="Validation Metrics",
        filename=os.path.join(experiment_dir, "val_metrics.jpeg"),
    )
