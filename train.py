# -*- coding: utf-8 -*-
import argparse
import glob
import os
import sys
import time
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils import hparams as hp
from Models.AttModel import AttModel
from Models.CTCModel import CTCModel
from utils.utils import frame_stacking, log_config, load_model, adjust_learning_rate, fill_variables, adjust_transformer_learning_rate
from Loss.label_smoothing import label_smoothing_loss
from load_waveform import load_waveform_model
import datasets

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_loop(model, optimizer, writer, step, args, hp) -> int:
    dataset_train = datasets.get_dataset(hp.train_script, hp, use_spec_aug=hp.use_spec_aug)
    if hp.encoder_type == 'Conformer':
        train_sampler = datasets.LengthsBatchSampler(dataset_train, hp.batch_size * 1500, hp.lengths_file)
        dataloader = DataLoader(dataset_train, batch_sampler=train_sampler, num_workers=2, collate_fn=datasets.collate_fn)
    else:
        train_sampler = DistributedSampler(dataset_train) if args.n_gpus > 1 else None
        dataloader = DataLoader(dataset_train, batch_size=hp.batch_size, shuffle=hp.shuffle, num_workers=2, sampler=train_sampler, collate_fn=datasets.collate_fn, drop_last=True)
    optimizer.zero_grad()
    for d in dataloader:
        step += 1
        if hp.encoder_type == 'Conformer':
            lr = adjust_transformer_learning_rate(step)
            print(f'step = {step}')
            print(f'lr = {lr}')
        text, mel_input, pos_text, pos_mel, text_lengths, mel_lengths = d

        text = text.to(DEVICE, non_blocking=True)
        mel_input = mel_input.to(DEVICE, non_blocking=True)
        pos_text = pos_text.to(DEVICE, non_blocking=True)
        pos_mel = pos_mel.to(DEVICE, non_blocking=True)
        text_lengths = text_lengths.to(DEVICE, non_blocking=True)

        if hp.frame_stacking > 1 and hp.encoder_type != 'Wave':
            mel_input, mel_lengths = frame_stacking(mel_input, mel_lengths, hp.frame_stacking)

        predict_ts = model(mel_input, mel_lengths, text, pos_mel)

        if hp.decoder_type == 'Attention':
            loss = label_smoothing_loss(predict_ts, text, text_lengths, hp.T_norm, hp.B_norm)
        elif hp.decoder_type == 'CTC':
            predict_ts = F.log_softmax(predict_ts, dim=2).transpose(0, 1)
            loss = F.ctc_loss(predict_ts, text, mel_lengths, text_lengths, blank=0)

        loss.backward()
        if step % hp.accum_grad == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
            optimizer.step()
            optimizer.zero_grad()
            loss.detach()

        if torch.isnan(loss):
            print('loss is nan')
            sys.exit(1)
        if hp.debug_mode == 'tensorboard':
            #writer.add_scalar("Loss/train", loss, step)
            print('loss = {}'.format(loss.item()))
        else:
            print('loss = {}'.format(loss.item()))

        sys.stdout.flush()
    return step

def train_epoch(model, optimizer, args, hp, start_epoch=0):
    writer = None
    if hp.load_wav_model:
        model = load_waveform_model(model, hp.load_wav_model)
        import pdb;pdb.set_trace()
        print(f'{hp.load_wav_model} loaded')
    dataset_train = datasets.get_dataset(hp.train_script, hp, use_spec_aug=hp.use_spec_aug)
    train_sampler = DistributedSampler(dataset_train) if args.n_gpus > 1 else None
    dataloader = DataLoader(dataset_train, batch_size=hp.batch_size, shuffle=hp.shuffle, sampler=train_sampler,
                            num_workers=1, collate_fn=datasets.collate_fn, drop_last=True)
    step = len(dataloader) * start_epoch

    for epoch in range(start_epoch, hp.max_epoch):
        start_time = time.time()
        step = train_loop(model, optimizer, writer, step, args, hp)
        if (epoch + 1) % hp.save_per_epoch == 0 or (epoch+1) % hp.reset_optimizer_epoch > 10:
            torch.save(model.state_dict(), hp.save_dir + "/network.epoch{}".format(epoch + 1))
            torch.save(optimizer.state_dict(), hp.save_dir + "/network.optimizer.epoch{}".format(epoch + 1))
        if hp.encoder_type != 'Conformer':
            adjust_learning_rate(optimizer, epoch + 1, hp)
        if (epoch + 1) % hp.reset_optimizer_epoch == 0:
        #if (epoch + 1) in hp.reset_optimizer_epoch == 0:
            if hp.encoder_type != 'Conformer':
                optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
        print("EPOCH {} end".format(epoch + 1))
        print(f'elapsed time = {(time.time()-start_time)//60}m')


def init_distributed(rank, n_gpus):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."

    torch.cuda.set_device(rank % n_gpus)

    os.environ['MASTER_ADDR'] = 'localhost' #dist_config.MASTER_ADDR
    os.environ['MASTER_PORT'] = '600010' #dist_config.MASTER_PORT

    torch.distributed.init_process_group(
        backend='nccl', world_size=n_gpus, rank=rank
    )


def cleanup():
    torch.distributed.destroy_process_group()


def run_distributed(fn, args, hp):
    try:
        mp.spawn(fn, args=(args, hp), nprocs=args.n_gpus, join=True)
    except:
        cleanup()

def run_training(rank, args, hp):
    if args.n_gpus > 1:
        init_distributed(rank, args.n_gpus)
        torch.cuda.set_device(f'cuda:{rank}')

    if hp.decoder_type == 'Attention':
        model = AttModel(hp)
    elif hp.decoder_type == 'CTC':
        model = CTCModel(hp)

    model = model.to(rank)
    if args.n_gpus > 1:
        model = DDP(model, device_ids=[rank])

    if hp.encoder_type == 'Conformer':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=1e-5)

    load_epoch = 0
    if hp.load_checkpoints:
        if hp.load_checkpoints_epoch is None:
            path_list = glob.glob(os.path.join(hp.load_checkpoints_path, 'network.epoch*'))
            for path in path_list:
                epoch = int(path.split('.')[-1].replace('epoch', ''))
                if epoch > load_epoch:
                    load_epoch = epoch
        else:
            load_epoch = hp.load_checkpoints_epoch
        print("{} epoch {} load".format(hp.load_checkpoints_path, load_epoch))
        model.load_state_dict(load_model(os.path.join(hp.load_checkpoints_path, 'network.epoch{}'.format(load_epoch))))
        if hp.is_flatstart:
            load_epoch = 0
            #pass
        else:
            optimizer.load_state_dict(torch.load(os.path.join(hp.load_checkpoints_path, 'network.optimizer.epoch{}'.format(load_epoch))))

    train_epoch(model, optimizer, args, hp, start_epoch=load_epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py')
    args = parser.parse_args()

    hp.configure(args.hp_file)
    fill_variables(hp)
    os.makedirs(hp.save_dir, exist_ok=True)
    writer = SummaryWriter(f'{hp.save_dir}/logs/{hp.comment}')

    log_config(hp)

    n_gpus = torch.cuda.device_count()
    args.__setattr__('n_gpus', n_gpus)

    if n_gpus > 1:
        run_distributed(run_training, args, hp)
    else:
        run_training(0, args, hp)
