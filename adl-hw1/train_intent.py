import json
import pickle
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import shuffle
from typing import Dict
import math
import torch
from tqdm import trange, tqdm

from intent_dataset import SeqClsDataset
from utils import Vocab
from intent_model import SeqClassifier
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    # {train, train.json}, {vel, eval.json}
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    # not yet convert to tensor idx
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets #
    train_loader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, shuffle=True, collate_fn = datasets[TRAIN].collate_fn)
    valid_loader = DataLoader(datasets[DEV], batch_size=args.batch_size, shuffle=False, collate_fn = datasets[DEV].collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu) # model yet
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets[TRAIN].num_classes, args.device)
    model.to(args.device)
    # TODO: init optimizer #
    optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)
    # shape {b, f, s}
    criterion = nn.CrossEntropyLoss()
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    #min_valid_loss = math.inf
    best_acc = 0.
    for epoch in epoch_pbar:
        train_acc = 0.
        valid_acc = 0.
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss = []
        for batch, data in enumerate(tqdm(train_loader, total=len(train_loader))):
            #print(data['text'].size(), data['intent'].size())
            data, label = data['text'].to(args.device), data['intent'].to(args.device)
            #print('start prediction')
            pred = model(data)
            optimizer.zero_grad()
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().item())
            output = torch.max(pred, 1)[1]
            train_acc += (output.cpu() == label.cpu()).sum().item()
            #print('epoch : {}, batch : {}, loss : {}, mean_train_loss : {}'.format(
            #    epoch + 1, batch + 1, loss.detach().item(), sum(train_loss) / len(train_loss)))
        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        valid_loss = []
        for batch, data in enumerate(tqdm(valid_loader)):
            #print('The shape of validating data = {}'.format(data['text'].shape))
            #print(data['text'].size(), data['intent'].size())
            data, label = data['text'].to(args.device), data['intent'].to(args.device)
            with torch.no_grad():
                pred = model(data)
                loss = criterion(pred, label)
                output = torch.max(pred, 1)[1]
                valid_acc += (output.cpu() == label.cpu()).sum().item()
            valid_loss.append(loss.item())
            #print('epoch : {}, batch : {}, loss : {}, mean_valid_loss : {}'.format(
            #    epoch + 1, batch + 1, loss.item(), sum(valid_loss) / len(valid_loss)))
        # record the model with best validation loss
        if(valid_acc > best_acc):
            best_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'best.pt'))
            print('Saving model with acc {:.3f}...'.format(best_acc / len(valid_loader.dataset)))
        print('Finsish epoch {} with train_loss : {}, valid_loss : {}, train_acc : {}, valid_acc : {}'.format(
            epoch + 1, sum(train_loss) / len(train_loss), sum(valid_loss) / len(valid_loss), train_acc / len(train_loader.dataset), valid_acc / len(valid_loader.dataset)))
        pass

    # TODO: Inference on test set
    # don't have to do here

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=0.001)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=80)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
