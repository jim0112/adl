# README
使用以下指令
```python=
# 會將產生的checkpoint-27140放到./
#兩個arg分別為train_data和dev_data(為助教定義的檔案路徑)
python train.py train.jsonl public.jsonl 
```
之後再根據產生的model(checkpoint-27140)，配合test.py來做後續的summarization即可。