import os
import torch
import torch.multiprocessing as mp


from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from src.data.data_loader import DigiFace
from src.data.preprocess import split_data
from src.losses.cosface import CosFaceLoss
from src.models.partfVit import PartFVitWithLandmark
from src.models.concat import ConcatModelWithLoss


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "4500"
    os.environ["NCCL_DEBUG"] = "INFO"
    init_process_group(backend="nccl", init_method="env://")


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        gpu_id,
        save_every,
        checkpoint_path,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader

        self.gpu_id = gpu_id
        self.save_every = save_every
        self.checkpoint_path = checkpoint_path

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _run_batch(self, source, target):
        self.optimizer.zero_grad()
        y_score = self.model(source, target)
        print(y_score.shape)
        loss = self.criterion(y_score, target)
        print(f"Current Loss: {loss.item()}")
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def _run_epoch(self, epoch):
        b_sz = len(self.train_loader)
        print(
            f"[GPU {self.gpu_id}] | Epoch: {epoch+1} | batchsize: {b_sz} | steps: {len(self.train_loader)}"
        )
        for i, (source, targets) in enumerate(self.train_loader):
            source, targets = source.to(self.gpu_id), targets.to(self.gpu_id)
            self._run_batch(source=source, target=targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        torch.save(ckp, self.checkpoint_path)
        print(f"epoch {epoch+1}: Training checkpoint saved at {self.checkpoint_path}")

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self._run_epoch(epoch)
            if str(self.gpu_id) == "0" and (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch)
        if str(self.gpu_id) == "0" and num_epochs % self.save_every != 0:
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

    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        pin_memory=True,
        sampler=DistributedSampler(train_data, shuffle=True),
    )

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
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

    return kargs


def start_proc(rank, world_size, config, experiment_dir):
    """
    Parameters
    ----------
    config : dict
            Configuration dictionary containing all the parameters for training
    experiment_dir : str
            Path to the experiment directory where model results, states and result will be saved
    """

    ddp_setup(rank, world_size)
    kargs = load_train_obj(config=config)

    trainer = Trainer(
        **kargs,
        save_every=config["save_every"],
        gpu_id=rank,
        checkpoint_path=config["save_path"],
    )

    trainer.train(num_epochs=config["num_epochs"])
    destroy_process_group()


def main(config, experiment_dir):
    world_size = torch.cuda.device_count()
    mp.spawn(start_proc, args=(world_size, config, experiment_dir), nprocs=world_size)
    destroy_process_group()
