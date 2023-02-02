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

train_dir = str(sys.argv[1])
sample_dir = str(sys.argv[2])
train_data = [json.loads(line) for line in open(train_dir, 'r')]
valid_data = [json.loads(line) for line in open(sample_dir, 'r')]

document = []
summary = []
id = []
for train in train_data:
  document.append(train['maintext'])
  summary.append(train['title'])
  id.append(train['id'])

train = dict({
    'document': document,
    'summary': summary,
    'id': id,
})

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
train_dataset = Dataset.from_dict(train)
valid_dataset = Dataset.from_dict(valid)

from datasets import load_dataset, load_metric
metric = load_metric("rouge")
from transformers import MT5Model, T5Tokenizer

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
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

train_datasets = train_dataset.map(preprocess_function, batched=True)
valid_datasets = valid_dataset.map(preprocess_function, batched=True)
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
batch_size = 8
args = Seq2SeqTrainingArguments(
    output_dir = "./",
    evaluation_strategy = "epoch",
    save_strategy="steps",
    save_steps= 27140,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    gradient_accumulation_steps = 2,
    save_total_limit = 1,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
)

import nltk
import numpy as np
nltk.download('punkt')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_datasets,
    eval_dataset=valid_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
