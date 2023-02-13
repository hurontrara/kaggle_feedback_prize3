import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from data_loader_cv import TextClassificationDataset, TextClassificationCollator
from utils_cv_ import read_text, Model, seed_everything, AWP
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm

import wandb

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--name', type=str, required=True)
    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_data_name', default='train_fold') # train_data
    p.add_argument('--pretrained_model_name', type=str, default="microsoft/deberta-v3-base")
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--adam_epsilon', type=float, default=1e-6)
    p.add_argument('--valid_fold', type=int, default=0)
    p.add_argument('--use_radam', action='store_true')
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--encoder_lr', type=float, default=2e-5)
    p.add_argument('--decoder_lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=.01)
    p.add_argument('--betas', type=float, default=(0.9, 0.999))
    p.add_argument('--max_grad_norm', type=float, default=1000)
    p.add_argument('--iteration_per_update', type=int, default=32)

    p.add_argument('--cosine', action='store_true')
    p.add_argument('--mse', action='store_true')

    p.add_argument('--MeanPooling', action='store_true')
    p.add_argument('--MultilayerCLSpooling', action='store_true')
    p.add_argument('--MultilayerMeanpooling', action='store_true')
    p.add_argument('--MultilayerWeightpooling', action='store_true')
    p.add_argument('--ConcatenatePooling', action='store_true')
    p.add_argument('--AttentionPooling', action='store_true')

    p.add_argument('--awp', action='store_true')

    config = p.parse_args()

    return config

def get_loaders(fn, tokenizer, valid_fold, config):

    train_texts, valid_texts, train_labels, valid_labels = read_text(fn, valid_fold, config)

    train_loader = DataLoader(
        TextClassificationDataset(train_texts, train_labels, config),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )

    valid_loader = DataLoader(
        TextClassificationDataset(valid_texts, valid_labels, config),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )

    return train_loader, valid_loader
           # index_to_label



def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    config.tokenizer = tokenizer

    train_loader, valid_loader = get_loaders(
        config.train_data_name,
        tokenizer,
        config.valid_fold,
        config
    )
    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    n_total_iterations = int(len(train_loader) / config.iteration_per_update * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)

    config.n_total_iterations = n_total_iterations
    config.n_warmup_steps = n_warmup_steps

    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )
# get model


    model = Model(config).cuda(config.gpu_id)

    # wandb.watch(model)


    # lr_monitor = LearningRateMonitor(logging_interval='step')
    wandb_logger = WandbLogger(project='feedback3', name='{}({}fold)'.format(config.name, config.valid_fold))
    wandb_logger.experiment.config.update(config)

    trainer = Trainer(max_epochs=config.n_epochs,
                      enable_progress_bar=False,
                      logger=wandb_logger,
                      gpus=1,
                      accelerator='gpu',
                      enable_checkpointing=False
                      )

    # precision=16,
    # accumulate_grad_batches=config.iteration_per_update,
    # callbacks = [lr_monitor, TQDMProgressBar()],

    trainer.fit(model, train_loader, valid_loader)

    # torch.save({
    #     'rnn' : None,
    #     'cnn' : None,
    #     'bert' : model.state_dict(),
    #     'config' : config,
    #     'vocab' : None,
    #     'classes' : index_to_label,
    #     'tokenizer' : tokenizer
    # }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    # wandb.init(project="Kaggle_FeedBack3", entity="hurontrara", name='{}'.format(config.name + '_' + str(config.valid_fold)))

    # wandb.config.update(config)
    num = random.randrange(1, 100)
    print('seed : {}'.format(num))
    seed_everything(seed=num)


    main(config)