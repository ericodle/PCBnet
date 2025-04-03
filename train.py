from tqdm import tqdm
from time import sleep
from utils.dataset import CocoDataset
from utils.model import create_model
from utils.training_utils import SaveBestModel, train_one_epoch, val_one_epoch, get_datasets
import torch
import os
import time
from simple_parsing import ArgumentParser
from eval import evaluate_model
from torch import nn
from torch.utils.tensorboard import SummaryWriter

def train(train_dataset, val_dataset, epochs, batch_size, exp_folder, val_eval_freq):
    date_format = "%d-%m-%Y-%H-%M-%S"
    date_string = time.strftime(date_format)
    exp_folder = os.path.join(exp_folder, "summary", date_string)
    writer = SummaryWriter(exp_folder)

    def custom_collate(data):
        return data

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
        pin_memory=torch.cuda.is_available()
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(train_dataset.get_total_classes_count() + 1).to(device)

    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    optimizer = torch.optim.SGD(pg0, lr=0.001, momentum=0.9, nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": 5e-4})
    optimizer.add_param_group({"params": pg2})

    save_best_model = SaveBestModel(output_dir=exp_folder)

    for epoch in range(epochs):
        model, optimizer, writer, epoch_loss = train_one_epoch(model, train_dl, optimizer, writer, epoch + 1, epochs, device)
        sleep(0.1)

        if epoch % val_eval_freq == 0 and epoch != 0:
            sleep(0.1)
        else:
            writer, val_epoch_loss = val_one_epoch(model, val_dl, writer, epoch + 1, epochs, device, log=True)
            sleep(0.1)
            save_best_model(val_epoch_loss, epoch, model, optimizer)

    _, _ = val_one_epoch(model, val_dl, writer, epochs, epochs, device, log=False)
    writer.add_hparams({"epochs": epochs, "batch_size": batch_size}, {"Train/total_loss": epoch_loss, "Val/total_loss": val_epoch_loss})

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_image_dir", type=str, required=True)
    parser.add_argument("--val_image_dir", type=str, required=True)
    parser.add_argument("--train_coco_json", type=str, required=True)
    parser.add_argument("--val_coco_json", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--val_eval_freq", type=int, default=2)
    parser.add_argument("--exp_folder", type=str, default="exp")
    args = parser.parse_args()

    train_ds, val_ds = get_datasets(
        train_image_dir=args.train_image_dir, train_coco_json=args.train_coco_json,
        val_image_dir=args.val_image_dir, val_coco_json=args.val_coco_json
    )
    train(train_ds, val_ds, args.epochs, args.batch_size, args.exp_folder, args.val_eval_freq)
