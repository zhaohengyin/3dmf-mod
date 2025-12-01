import os 
from pathlib import Path
import numpy as np 
import json
import copy
import wandb 
import torch.distributed as dist
import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.cuda.amp import autocast, GradScaler 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset, ConcatDataset, Dataset
import torch.multiprocessing as mp
from mf_module.model import MotionDetectionModelBuilder
from mf_module.model.ema import EMAModel
from mf_module.data.dataset import MotionDetectionDataset
from mf_module.data import MotionDetectionParserBuilder
from mf_module.utils.json_utils import dump_dict_to_json
import time 
import signal 


# ----------------------------------------
# Data and Model Builder
def build_dataset(args):
    builder = MotionDetectionParserBuilder()
    parser = builder.build(args)
    dataset = MotionDetectionDataset(Path("./asset/dataset/motion_dataset/"), parser)
    return dataset


def build_model(args):
    builder = MotionDetectionModelBuilder()
    return builder.build(args)


# ----------------------------------------
# Trainer Function

def train(
    root_path, 
    model, 
    dataloader, 
    args, 
    test_dataloader=None, 
    rank=0, 
    local_rank=0, 
    optim_dict=None, 
    alpha_v=3.0, 
    ema_power=0.66,
    total_epoch=10000,
    save_per_epoch=20,
    **kwargs
):
    assert save_per_epoch > 0 and isinstance(save_per_epoch, int)
    assert alpha_v >= 0.0

    model.train()
    ema_model = copy.deepcopy(model)
    ema_model.load_state_dict(model.state_dict())
    ema = EMAModel(ema_model, power=ema_power)

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args["lr"], 
        weight_decay=args["weight_decay"]
    )

    if optim_dict is not None:
        optimizer.load_state_dict(optim_dict)
    
    # Counters
    training_step = 0
    agg_loss = 0

    last_time = time.time()

    huber_v = nn.SmoothL1Loss(beta=0.3, reduction='none')
    huber_depth = nn.SmoothL1Loss(beta=0.003, reduction='none') 

    for epoch in range(1, 1+ total_epoch):
        agg_loss = 0.0
        for idx, batch in enumerate(dataloader): 
            x = batch["x"]
            cam = batch["cam"].to(local_rank)
            cam_t = batch["cam_t"].to(local_rank)

            mask = batch["mask"].to(local_rank)
            y = batch["y"].to(local_rank)

            if isinstance(x, dict):
                for k, v in x.items():
                    x[k] = v.to(local_rank)
                    print(k, v.shape)
            else:
                x = x.to(local_rank)

            # Forward
            y_pred = model.forward(x, cam, cam_t)["all"]

            # Compute Loss
            coef = x.shape[0] * x.shape[2] * x.shape[3] / (mask.sum() + 0.001)
            loss_v = (mask * (huber_v(y[:, 1:], y_pred[:, 1:]))).mean() * coef
            loss_depth = (mask * (huber_depth(y[:, :1], y_pred[:, :1]))).mean() * coef 
            loss = loss_depth + loss_v * alpha_v

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            training_step += 1
            
            # Log
            agg_loss += loss.item()

            if training_step % args["n_save_step"] == 0 and rank == 0:
                torch.save(model.state_dict(), f"{root_path}/last.pth")
                torch.save(optimizer.state_dict(), f"{root_path}/last_opt.pth")
        
        print(f"Epoch {epoch} loss", agg_loss / (idx + 1))
        if rank == 0:
            torch.save(model.state_dict(), f"{root_path}/last.pth")
            torch.save(ema_model.state_dict(), f"{root_path}/last_ema.pth")
            torch.save(optimizer.state_dict(), f"{root_path}/last_opt.pth")

        if epoch % save_per_epoch == 0 and rank == 0:
            torch.save(model.state_dict(), f"{root_path}/epoch_{epoch}.pth")
            torch.save(ema_model.state_dict(), f"{root_path}/epoch_{epoch}_ema.pth")
            torch.save(optimizer.state_dict(), f"{root_path}/epoch_{epoch}_opt.pth")


def main():  
    rank = local_rank = 0
    arg_dict = {
        "exp_tag": "exp_motion",
        "tag": "motion_model",
        "training": {
            "lr": 0.00003,
            "weight_decay": 0.01,
            "bs": 4,
            "n_save_step": 2500
        },

        "dataset": {
            "type": "default",
            "args": {
                "port": 0
            }
        },

        "model": {
            "type": "default",
            "args": {
                "dropout": True,
                "dropout_val": 0.01,
                "n_base_channel": 64
            }
        },

        "checkpoint": ""
    }

    # Setup Model and Dataset
    dataset = build_dataset(arg_dict["dataset"])
    model = build_model(arg_dict["model"]).to(local_rank)

    optim_dict = None
    if arg_dict["checkpoint"] != "":
        model.load_state_dict(torch.load(Path("output") / arg_dict["checkpoint"] / "last.pth"))
        optim_dict = torch.load(Path("output") / arg_dict["checkpoint"] / "last_opt.pth")
    
    # Setup Distributed Dataloader
    base_batchsize = arg_dict["training"]["bs"]
    
    dataloader = DataLoader(dataset, 
        batch_size=base_batchsize, 
        shuffle=True, 
        drop_last=False, 
        num_workers=8, 
        prefetch_factor=1,
        pin_memory=False, 
        persistent_workers=False,
    )
    
    # Initialize Training Context 
    if rank == 0:
        model_name = arg_dict["tag"]
        root_path = Path(f"./output/{model_name}")
        os.makedirs(root_path)
        dump_dict_to_json(arg_dict, root_path / "config.json")

    else:
        root_path = None

    # Launch Training.
    train(
        root_path,
        model, 
        dataloader, 
        args=arg_dict["training"], 
        test_dataloader=None, 
        rank=rank, 
        local_rank=local_rank, 
        optim_dict=optim_dict
    )


if __name__ == '__main__':
    main()
