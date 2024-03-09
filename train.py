import yaml
import torch
import torch.nn.functional as F

from src.data.data_set import DigiFace
from src.models.partfvit import PartfVit


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device).squeeze(dim=1)
        output = model(input, label)
        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(input),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )
    with open("grads.txt", "a") as f:
        for n, p in model.named_parameters():
            if p.grad is None:
                f.write(f"{n} {None}\n")
            elif p.grad.norm() < 1e-4:
                f.write(f"{n} {p.grad.norm()}\n")


def valid(model, device, val_loader, criterion):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for input, label in val_loader:
            input, label = input.to(device), label.to(device).squeeze(dim=1)
            output = model(input, label)
            valid_loss += criterion(output, label).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    valid_loss /= len(val_loader.dataset)

    print(
        "\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            valid_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return valid_loss


def prepare_objects(config):
    train_dataset = DigiFace(config["data_path"], "train")
    valid_dataset = DigiFace(config["data_path"], "valid")

    model = PartfVit(
        num_ids=train_dataset.num_ids,
        image_size=config["image_size"],
        mobilenet_size=config["mobilenet_size"],
        num_landmarks=config["num_landmarks"],
        patch_size=config["patch_size"],
        mlp_dim=config["mlp_dim"],
        feat_dim=config["feat_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        eps=1e-10,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config["T_0"], T_mult=config["T_mult"], eta_min=config["eta_min"]
    )

    criterion = torch.nn.CrossEntropyLoss()

    return model, criterion, optimizer, scheduler, train_dataset, valid_dataset


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model, criterion, optimizer, scheduler, train_dataset, valid_dataset = (
        prepare_objects(config)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=False
    )

    model = model.to(device)

    for epoch in range(1, config["epochs"] + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        valid_loss = valid(model, device, valid_loader, criterion)
        scheduler.step(valid_loss)
        torch.save(model.state_dict(), "checkpoint.pth")


if __name__ == "__main__":
    main()
