from datetime import datetime

from src.losses import combined_loss
from src.metrics import dice_score
from src import trainUtil
from src.model import SC_Net

from src.dataloader import Dataset
from src.augmentation import (
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing,
)
import config

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(train_dir, test_dir, dataset_name):

    model = SC_Net(
        device=config.device,
    )

    preprocessing_fn = None

    train_dataset = Dataset(
        train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        mode="train",
    )

    valid_dataset = Dataset(
        test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        mode="test",
    )

    batch_size = config.batch_size #32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=32,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=32,
    )

    metrics = [
        dice_score,
    ]

    optimizer = torch.optim.RAdam([#Rectified Adam
        dict(params=model.parameters(),
             lr=config.learning_rate,
             betas=(0.9, 0.999)),
    ])


    torch.autograd.set_detect_anomaly(True)

    loss_fn = combined_loss
    train_epoch = trainUtil.TrainEpoch(
        model,
        loss=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        device=config.device,
        verbose=True,
    )

    valid_epoch = trainUtil.ValidEpoch(
        model,
        loss=loss_fn,
        metrics=metrics,
        device=config.device,
        verbose=True,
    )

    writer_path = "./{}/SC-Net_{}".format(config.tensorboard_logs,dataset_name)
    writer = SummaryWriter(writer_path)
    min_loss = 9999
    max_score = 0
    last_save = 0


    for i in range(1, config.epochs):

        print("\nEpoch: {}".format(i))
        start_time = datetime.now()
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        writer.add_scalar("Loss/train", train_logs[loss_fn.__name__], i)
        writer.add_scalar("Loss/valid", valid_logs[loss_fn.__name__], i)
        writer.add_scalar("Dice/train", train_logs[metrics[0].__name__], i)
        writer.add_scalar("Dice/valid", valid_logs[metrics[0].__name__], i)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], i)

        saved = False
        if min_loss > valid_logs[loss_fn.__name__]:
            min_loss = valid_logs[loss_fn.__name__]
            torch.save(
                model.state_dict(),
                "./{}/weight.pth".format(config.checkpoints_dir)
            )
            last_save = i
            print("Model saved Loss!")
            saved = True
        if max_score < valid_logs[metrics[0].__name__] and saved == False:
            max_score = valid_logs[metrics[0].__name__]
            torch.save(
                model.state_dict(),
                "./{}/weight.pth".format(config.checkpoints_dir)
            )
            last_save = i
            print("Model saved Metric!")

        if i - last_save >= 80:
            last_save = i
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.5
            print("Decrease decoder learning rate to ",
                  optimizer.param_groups[0]["lr"])

        end_time = datetime.now()
        cost_time = end_time - start_time
        print("Cost time:{},time left:{}".format(cost_time,cost_time*(config.epochs-i)))

    writer.flush()


if __name__ == "__main__":
    dataset_name = "consep"
    train_dir = "data/{}/toy_train/".format(dataset_name)
    test_dir = "data/{}/toy_test/".format(dataset_name)
    train(train_dir, test_dir, dataset_name)
