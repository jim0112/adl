import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import DataLoader

from slot_dataset import SeqClsDataset
from slot_model import SeqClassifier
from utils import Vocab
import csv

def predict(loader, model, device):
    model.eval()
    model.to(device)
    preds = []
    for data in loader:
        data = data['tokens'].to(device)
        with torch.no_grad():                   
            output = model(data)
            pred = torch.max(output, 2)[1]
            preds.append(pred.detach().cpu())
    # 32 -> 1   
    preds = torch.cat(preds, dim=0).numpy()
    return preds

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"

    # making <-> dict
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    idx2tag = {idx: intent for intent, idx in tag2idx.items()}

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn = dataset.collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        args.device
    )
    # load weights into model
    model.load_state_dict(torch.load(args.ckpt_path))

    # TODO: predict dataset
    preds = predict(test_loader, model, args.device)
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tags'])
        for i, pred in enumerate(preds):
            # if x == 9, then that is a pad
            trim = list(filter(lambda x: (x != 9), pred.tolist()))
            record = ' '.join(list(map(lambda x: idx2tag[x], trim)))
            writer.writerow(['test-{}'.format(i), record])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/best.pt"
    )
    parser.add_argument("--pred_file", type=Path, default="pred_slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
