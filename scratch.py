from transformers import GPT2Tokenizer, GPT2Model
import torch
import matplotlib.pyplot as plt
import numpy as np


def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            labels.append(line[:-1])

    return labels

def convert_to_weighted_label(val, alpha):

    weighted_label = []

    return weighted_label


def get_difference(idx, labels, vals):

    diff_all = []
    for i in range(len(vals)):
        diff = np.abs(vals[idx] - vals[i])
        diff = np.sum(diff)
        diff_all.append(diff)

    return np.array(diff_all)




labels = load_labels('imagenet_labels.txt')
labels = np.array(labels)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = 'tiger'

encoded_input = tokenizer(text, return_tensors='pt', max_length=50, truncation=True)
output = model(**encoded_input)


vals = []
for l in labels:
    length = len(l.split(' '))
    text = 'Describe {}: It looks like'.format(l)
    encoded_input = tokenizer(text, return_tensors='pt', max_length=50, truncation=True)

    with torch.no_grad():
        output = model(**encoded_input)

    val = torch.squeeze(output.last_hidden_state)

    val = val.detach().numpy()
    vals.append(val[-1])


labels = np.array(labels)
vals = np.array(vals)

for i in range(1000):
    diff = get_difference(i, labels, vals)
    args = np.argsort(diff)
    print(labels[i], labels[args[:5]])


exit()