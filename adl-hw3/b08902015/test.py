import json
import tqdm
import numpy as np
import random
import sys
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, T5TokenizerFast, MT5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
class T5_Dataset(Dataset):
    def __init__(self, data):
      self.ids = data['input_ids']
      self.title = data['labels']
      self.sum = data['summary']
    def __getitem__(self, idx):
        return torch.as_tensor(self.ids[idx]), self.sum[idx]
    
    def __len__(self):
        return len(self.ids)
model_dir = './checkpoint-27140'
sample_dir = str(sys.argv[1])
out_file = str(sys.argv[2])
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
valid_data = [json.loads(line) for line in open(sample_dir, 'r')]
document = []
summary = []
id = []
for valid in valid_data:
  document.append(valid['maintext'])
  summary.append(valid['title'])
  id.append(valid['id'])

valid = dict({
    'document': document,
    'summary': summary,
    'id': id,
})

from datasets import Dataset
valid_dataset = Dataset.from_dict(valid)

max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    inputs = ['summarize: ' + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

valid_datasets = valid_dataset.map(preprocess_function, batched=True)
dev_dataset = T5_Dataset(valid_datasets)
dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False, pin_memory=True)
#from tw_rouge import get_rouge
import tqdm
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
res = []
title = []
batch = 32
model.eval()
'''
greedy
num_beams= 5 or 10
top_k = 10 or 20
top_p = 0.92 or 0.7
temperature = 0.5 or 2
'''
for p, q in tqdm.tqdm(dev_dataloader):
    p = p.to(device)
    output = tokenizer.batch_decode(model.generate(p, num_beams=5, max_length=128, repetition_penalty=3.0, do_sample=False),skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(output)
    res.extend(output)
#rouge = get_rouge(res, title)
#print(rouge)

with open(out_file, "w", encoding="utf8") as fp:
  i = 21710
  for r in res:
    json.dump({"title": r, "id": "{}".format(i)}, fp, ensure_ascii = False)
    i += 1
    fp.write("\n")



