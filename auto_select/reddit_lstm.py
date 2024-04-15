import argparse
import os

from pickletools import optimize
import random
import string
import time
from math import log
import numpy as np
import scipy.sparse as sp
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
from torch import Tensor, nn
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
import pickle as pkl
import json
from tools import utils
from ordered_set import OrderedSet


# 读取进行了文本数据清洗的数据，pkl文件
def get_cleaned_data():
    with open('../data/reddit_clean.pkl', 'rb') as f:
        data = pkl.load(f)
    return data


# 将reddit_json中的subreddit中set集合转化为列表
def contents_elements(data_set):
    # 验证数据类型
    # print(f"Data type: {type(data_set)}")
    # 如果是集合，则转换为列表并打印
    if isinstance(data_set, set):
        data_list = list(data_set)
        return data_list
    else:
        print("Data is not a set.")



class EarlyStopping(object):

    def __init__(self, num_trials,path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = path

    def is_continuable(self, model, accuracy):


        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save({'state_dict': model.state_dict()},self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False
        


# 读取数据,得到文本的语料库
def get_corpus():
    data = get_cleaned_data()

    # 词汇表，存储在Set集合中，并非是按照push的顺序进行的。

    word_set =OrderedSet()  # 谁在文档中出现过谁第一
    # 遍历reddit_pkl中的subreddit，将其中的set集合转换为列表
    for i in range(len(data)):
        for j in range(len(data[i]['subreddit'])):
            contents = data[i]['subreddit'][j].split()
            for word in contents:
                # print(word)
                word_set.add(word)
    vocab = list(word_set) # 词汇表
    vocab_size = len(vocab) # 词汇表大小
    word_id_map = {}
    id_word_map = {}
    # 词汇表的索引
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i
        id_word_map[i] = vocab[i]
    return vocab, word_id_map, id_word_map


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = np.array(data)  # 将数据转换为 NumPy 数组
        self.labels = np.array(labels)  # 将标签转换为 NumPy 数组
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)




class LSTM_classifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_labels, dropout) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,num_layers=1, batch_first=True, dropout=dropout, bidirectional=False)
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, inputs):
        emb = self.embedding(inputs)
        # print(emb.shape)
        output, (h_n, c_n) = self.lstm(emb)

        # todo 取注意力机制/或者进行策略选择评分



        inter_output = torch.mean(output, dim=1)
        # todo 进行注意力机制
        res = self.classifier(inter_output)
        # 每个词的输出
        return output, res
    
  
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    # 窗口大小
    parser.add_argument('--window_size', type=int, default=20)
    # parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_size", type=int, default=200)
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args(args)

# 将词汇库转化为ID
def trans_corpus_to_ids(corpus, word_id_map, max_len):
    # 新的语料库
    new_corpus = []

    # print(word_id_map)
    for text in corpus:
        # print(text)
        word_list = text.split() # 将文本分割成词
        if len(word_list) > max_len:
            word_list = word_list[:max_len]
            # print(word_list)
        new_corpus.append([word_id_map[w] + 1 for w in word_list])
    
    for i, one in enumerate(new_corpus):
        if len(one) < max_len:
            new_corpus[i] = one + [0]*(max_len-len(one))
    new_corpus = np.asarray(new_corpus, dtype=np.int32)
    return new_corpus

# 评估LSTM模型
def lstm_eval(model, dataloader, device):
    model.eval()
    all_preds, all_labels,all_outs = [],[],[]
    for batch in dataloader:
        batch = [one.to(device) for one in batch]
        x, y = batch
        with torch.no_grad():
            output, pred = model(x)
            # 预测的结果
            all_outs.append(output.cpu().numpy())
            pred_ids = torch.argmax(pred, dim=-1)
            all_preds += pred_ids.tolist()
            all_labels += y.tolist()
    # 计算准确率，即预测正确的数量除以总的数量
    acc = np.mean(np.asarray(all_preds) == np.asarray(all_labels))
    # 将所有的输出拼接在一起，按行拼接
    all_outs = np.concatenate(all_outs, axis=0)

    model.train()
    return acc, all_outs



def train_lstm(word_id_map, emb_size, hidden_size, dropout, batch_size, epochs, lr, weight_decay, num_labels,device,max_len):
    vocab_size = len(word_id_map) + 1 # padding填充为0
    reddit_label = []
    reddit_contents = []
    reddit_pkl = get_cleaned_data()
    for i in range(len(reddit_pkl)):
        reddit_label.append(reddit_pkl[i]['label'])
        # reddit_pkl[i]['label']内容进行拼接
        contents = ''
        for j in range(len(reddit_pkl[i]['subreddit'])):
            contents += reddit_pkl[i]['subreddit'][j] + ' '
        reddit_contents.append(contents)


    # todo BUG 每次加载的数据不一致    SET() 集合的问题
    corpus_ids = trans_corpus_to_ids(reddit_contents, word_id_map, max_len)



    train_data, test_data, train_labels, test_labels = train_test_split(corpus_ids, reddit_label, test_size=0.2, random_state=42,stratify=reddit_label)

    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,  batch_size, shuffle=False)

    is_early_stopping = EarlyStopping(epochs, path='./trained_model/lstm_model.pth')


    model = LSTM_classifier(vocab_size, emb_size, hidden_size, num_labels, dropout)
    model.to(device)
    # 训练数据集
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_func = nn.CrossEntropyLoss(reduction='mean')
    for ep in range(epochs):
        model.train()
        for batch in tqdm(train_loader):
            batch = [one.to(device) for one in batch]
            x, y = batch 
            output, pred = model(x)
            # loss = loss_func(pred, y)
            loss = utils.loss_function(pred, y, loss_type='ce')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch: %d, Loss: %.5f' % (ep, loss.item()))
        model.eval()
        correct = 0
        total = 0
        for data, labels in test_loader:
            data, labels = data.cuda(), labels.cuda()
            _,res = model(data)
            # print(outputs,'in test')
            _, predicted = torch.max(res.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('Accuracy: %.3f %%' % accuracy)
        if not is_early_stopping.is_continuable(model, accuracy):
            break


    # 评估模型
    model.load_state_dict(torch.load(is_early_stopping.save_path)['state_dict'])
    model.eval()
    correct = 0
    total = 0
    fin_targets = []
    fin_outputs = []
    for data, labels in test_loader:
        data, labels = data.cuda(), labels.cuda()
        outputs,res = model(data)
        _, predicted = torch.max(res.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        fin_targets.extend(labels.cpu().detach().numpy().tolist())
        fin_outputs.extend(predicted.cpu().detach().numpy().tolist())

    fin_outputs =  np.hstack(fin_outputs)
    fin_targets =  np.hstack(fin_targets)
    M = utils.gr_metrics(fin_outputs, fin_targets)
    print('GP:', M[0])
    print('GR:', M[1])
    print('FS:', M[2])

    precision_score1 = precision_score(fin_targets, fin_outputs,average='weighted')
    recall_score1 = recall_score(fin_targets, fin_outputs,average='weighted')
    f1_score1 = f1_score(fin_targets, fin_outputs,average='weighted')
    print('Precision:', precision_score1)
    print('Recall:', recall_score1)
    print('F1:', f1_score1)

    accuracy = 100 * correct / total
    print('Final accuracy: %.3f %%' % accuracy)

def set_seed(args):
    """
    :param args:
    :return:
    """
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # torch.backends.cudnn.deterministic = True


def main():
   


    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reddit_data = get_corpus()
    _, word_id_map, _ = reddit_data

    set_seed(args)
    # 训练LSTM模型
    train_lstm(word_id_map, args.embed_size, args.hidden_size, args.dropout, args.batch_size, args.epochs, args.lr, args.weight_decay, 5,device,args.max_len)


# todo 每次结果不一致的问题是由于采样的数据不一致导致的，可以通过设置随机种子解决



if __name__ == '__main__':
    main()