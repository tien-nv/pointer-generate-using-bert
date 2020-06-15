import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import data_util.config as config


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = '[PAD] [PAD]'
t = tokenizer.tokenize(text)
x = tokenizer.convert_tokens_to_ids(t)
y = [0]*len(x)
# keys = list(tokenizer.vocab.keys())

# values = list(tokenizer.vocab.values())
# with open('/home/vbee/tiennv/pointer_summarizer/dataset/vocab_bert','a+',encoding='utf-8') as f:
#     for tup in zip(keys, values):
#         f.write(str(tup[0]) + " " + str(tup[1]) + "\n")
a = torch.tensor([x])
b = torch.tensor([y])
en,_ = config.model(a,b)
print(en[11][0])