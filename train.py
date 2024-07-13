import argparse
import logging
import os

import torch
from torch import distributed
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset.dataset_loader import get_dataloader
from dcc import DCC
from learning.losses import CombinedMarginLoss
from utils.utils_callbacks import CallBackLogging
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import init_logging, AverageMeter

device = torch.device('cuda')


def main(args):
    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    os.makedirs(cfg.output, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))

    init_logging(0, cfg.output)

    train_loader = get_dataloader(
        batch_size=cfg.batch_size,
        roots=cfg.data_root,
        anno_files=cfg.anno_files,
        num_workers=cfg.num_workers,
        sample_num=cfg.sample_num,
        num_image=cfg.num_image
    )

    backbone_q = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    backbone_k = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    backbone_k.requires_grad_(False)
    backbone_q.train().cuda()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
    )

    module_dcc = DCC(backbone_q=backbone_q, backbone_k=backbone_k, margin_loss=margin_loss,
                     embedding_size=cfg.embedding_size, batch_size=cfg.batch_size,
                     fp16=False, queue_size=cfg.queue_size,
                     sample_num=cfg.sample_num)
    module_dcc.train().cuda()
    opt = torch.optim.SGD(
        params=[{"params": backbone_q.parameters()}],
        lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    lr_scheduler = CosineAnnealingLR(opt, T_max=cfg.total_step)

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    global_step = 0
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(0, cfg.num_epoch):
        for index, (img, local_labels) in enumerate(train_loader):
            opt.zero_grad()
            global_step += 1
            local_labels = local_labels.view(cfg.batch_size).long()

            img_q = img[0]
            img_k = img[1:]

            loss = module_dcc(img_q, img_k, local_labels.to(device), cfg)

            amp.scale(loss).backward()
            amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(backbone_q.parameters(), 5)
            amp.step(opt)
            amp.update()
            lr_scheduler.step()
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)
       
        path_module = os.path.join(cfg.output, f"model{epoch}.pt")
        torch.save(backbone_q.state_dict(), path_module)

    path_module = os.path.join(cfg.output, f"model{epoch}.pt")
    torch.save(backbone_q.state_dict(), path_module)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    local_rank = 0
    distributed.init_process_group(
        backend="gloo",
        init_method="tcp://localhost:12345",
        rank=0,
        world_size=1,
    )

    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", default='configs/MS1MV3',
                        type=str,
                        help="py config file")
    main(parser.parse_args())
