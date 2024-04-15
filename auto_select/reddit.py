import ast
import torch
import torch.nn as nn
import string
import nltk # 用于处理人类语言数据的符号和统计自然语言处理（NLP）。它提供了文本处理库用于分类、标记、语法分析、语义推理等。
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
# import tqdm, gc, time
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from torchfm.layer import CrossNetwork
from nltk.stem.wordnet import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from transformers import (AdamW, get_cosine_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup)
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from scipy.stats import zscore
import model_adfs as ADFS
from sklearn.metrics import accuracy_score
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from tools import utils



# 读取数据
reddit = utils.load_df('reddit_500')


# def clean_text(text):
#   punctuation = set(string.punctuation)
#   data = "".join(pos for pos in string_list(data[1]) if pos not in punctuation)


# # 定义一个 Min-Max 归一化的函数
def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())


def log_normalize(column):
    return np.log1p(column)

def string_list(post):
    # 将每个post转换为list
    converted_list = ast.literal_eval(post)
    return converted_list


# 提取统计句子，词汇特征
def get_satisc_features(data):
    num_pos_token,  num_pos_sent , num_pos_words = [], [], []
    for str in data['Post']:
        temp_token, temp_sent , temp_pos = 0, 0, 0 
        for pos in string_list(str):
          # 单词数
          temp_token += len(nltk.word_tokenize(pos))
          # 句子数
          temp_sent += len(nltk.tokenize.sent_tokenize(pos, language='english'))
          temp_pos += len(pos.split())
        num_pos_token.append(temp_token)
        num_pos_sent.append(temp_sent)
        num_pos_words.append(temp_pos)

    features = {'pos_token': num_pos_token,'pos_sent': num_pos_sent,'pos_words': num_pos_words}
    df =  pd.DataFrame(features, columns=['pos_token', 'pos_sent', 'pos_words'])
    # 对 DataFrame 的每列应用归一化
    zscore_df = df.apply(zscore)
    # normalized_df = df.apply(log_normalize)
    # normalized_df = df.apply(min_max_normalize)
    # normalized_df.to_csv('features.csv', index=False)
    # df.to_csv('features.csv', index=False)

    # return normalized_df
    return zscore_df
    # return df



# 得到词性标定（POS）
def get_all_tags(data):
    print("Processing POS features ...")
    tags_all = []
    punctuation1 = set(string.punctuation)
    for tags in data['Post']:
        for body in string_list(tags):
          body = "".join(pos for pos in body if pos not in punctuation1)
          tagged_text = nltk.pos_tag(nltk.word_tokenize(body))
          for word, tag in tagged_text:
              if tag not in tags_all:
                  tags_all.append(tag)
    return tags_all


# get_all_tags(reddit)
# 提供的数据集中提取词性标记（POS）的统计特征。
def f_pos(data, tags_all):
  # tag_count: 存储每个标题中各词性标记的出现次数。
  # tag_count_body: 存储每个正文中各词性标记的出现次数
  punctuation1 = set(string.punctuation)
  tag_dict, tag_count_body = {}, {}
  # print(len(tags_all))
  for tag in tqdm(tags_all):
      tag_dict[tag] = 0
      tag_count_body[tag] = []
  for posts in tqdm(data['Post']):
      for tag in tags_all:
          tag_dict[tag] = 0
      for text_sentence in string_list(posts):
        text_sentence = "".join(pos for pos in text_sentence if pos not in punctuation1)
        tagged_text = nltk.pos_tag(nltk.word_tokenize(text_sentence))
        for word, tag in tagged_text:
            tag_dict[tag] += 1
          # print(tag_dict.values(), tag_dict.keys(),105)
      for count, tag in zip(tag_dict.values(), tag_dict.keys()):
          # 将每个用户的pos向量进行记录
          tag_count_body[tag].append(count)

  pd.DataFrame(tag_count_body, index=None).to_csv('tag_count_body.csv', index=False)
  df = pd.DataFrame(tag_count_body, index=None)
#   log_df = df.apply(zscore)
  log_df = df.apply(log_normalize)
  # log_df = df.apply(min_max_normalize)
#   log_df.to_csv('log_tag_count_body.csv', index=False)
  return log_df
  # return df


# 提取if-idf特征
def get_ifidf_features(data):
    print("Processing TF-IDF features ...")
    X = []
    for t in data['Post']:
        X.append(t)
    # 这个对象用于将文本转换为词频矩阵。它配置了三个参数：移除英文停用词，只考虑单个词的n-grams，最多考虑频率最高的50个词。
    count_vect = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_features=50)
    X_counts = count_vect.fit_transform(X)
    word = count_vect.get_feature_names_out()

    # 将word词保存到CSV文件
    pd.DataFrame(word).to_csv('word.csv', index=False)

    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    # 将稀疏矩阵转换为稠密矩阵，并创建DataFrame
    df_tfidf = pd.DataFrame(X_tfidf.todense(), columns=word)
    # 保存DataFrame到CSV文件
    # df_tfidf.to_csv('tfidf_features.csv', index=False)
    # df_tfidf.to_csv('tfidf_features.csv', columns=word, index=False)  
    # return pd.DataFrame(X_tfidf.todense(), columns=word)
    return df_tfidf
    
  
  # 得到vadder的情绪分析


def get_emotion_features(data):

    print("Processing emotion features ...")
    # 创建 SentimentIntensityAnalyzer 对象
    analyzer = SentimentIntensityAnalyzer()

    sentiment = {
        'neg': [],
        'neu': [],
        'pos': [],
        'compound': []
    }

    for i in range(len(data['Label'])):
        # 获取情感分析结果
        sentiment_score = analyzer.polarity_scores(data['Post'][i])

        for key, value in sentiment_score.items():
            sentiment[key].append(value)
    df = pd.DataFrame(sentiment, index=None)
    # df.to_csv('emotion_features.csv', index=False)
    return df




# 得到自杀字典统计特征
def get_sd_features(data):
    print("Processing sd features ...")
    sd = utils.load_SD()
    # 创建自杀自杀字典
    sd_dict = {}
    sd_df = {}
    lemmatize = WordNetLemmatizer()
    for label in sd['lexicon']:
       if label not in sd_dict:
           label = lemmatize.lemmatize(label.lower())
           sd_dict[label] = 0
           sd_df[label] = []
      
    for posts in data['Post']:
        for post in string_list(posts):
            for word in nltk.word_tokenize(post):
                word = lemmatize.lemmatize(word.lower())
                if word in sd_dict:
                    sd_dict[word] += 1
            
        for k,v in sd_dict.items():
            sd_df[k].append(v)
            sd_dict[k] = 0
        

    df = pd.DataFrame(sd_df, index=None)
    # return df

    # 按照a、b、c、d进行类别上的统计
    # df.to_csv('sd_features.csv', index=False)

    indices_a = sd[sd['computer'].str.contains('a', na=False)].index
    indices_b = sd[sd['computer'].str.contains('b', na=False)].index
    indices_c = sd[sd['computer'].str.contains('c', na=False)].index
    indices_d = sd[sd['computer'].str.contains('d', na=False)].index

    sd_a = df.iloc[:, indices_a].sum(axis=1)
    sd_b = df.iloc[:, indices_b].sum(axis=1)
    sd_c = df.iloc[:, indices_c].sum(axis=1)
    sd_d = df.iloc[:, indices_d].sum(axis=1)

    sd_features = pd.DataFrame({'sd_a': sd_a, 'sd_b': sd_b, 'sd_c': sd_c, 'sd_d': sd_d})
    sd_features = sd_features.apply(log_normalize)
    # sd_features.to_csv('sd_ABCD_features.csv', index=False)
    return sd_features


if __name__ == '__main__':
    seed=43
    print(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    reddit = utils.load_df('reddit_500')
    
    # 提取统计句子，词汇特征
    satisc_data = get_satisc_features(reddit)
    # 提取词性标定（POS）
    
    tags_all = get_all_tags(reddit)
    # 提取词性标定（POS）
    pos_data = f_pos(reddit, tags_all)

    # 提取统计句子，词汇特征
    ifidf_data = get_ifidf_features(reddit)

    # 提取情感特征
    emotion_data = get_emotion_features(reddit)

    # sd_features = get_sd_features(reddit)


    # features = pd.concat([ pos_data,ifidf_data,emotion_data,], axis=1)
    features = pd.concat([ pos_data,ifidf_data,emotion_data,], axis=1)


    # features.to_csv('S_P_F_E_features.csv', index=False)

    # features.to_csv('./sdf.csv', index=False)
    # features.to_csv('P_F_E_features.csv', index=False)

    # features = pd.read_csv('./sdf.csv')


    
    inpus_dim = {
        # 'satisc_data': 3,

        'pos_data': 36,
        'ifidf_data': 50,
        'emotion_data': 4,
        # 'sd_features': 4,


    }

    num = len(inpus_dim)
    
    # 提取if-idf特征
    # features = get_ifidf_features(reddit)

    # 提取情感特征
    # features = get_emotion_features(reddit)

    df_all = pd.concat([features, reddit['Label']], axis=1)
    # df_all = pd.read_csv('./features_tf_50.csv')
    # df_all.to_csv('features_tf_50.csv', index=False)
    # print(df_all)
    # print(features.shape)

    X = df_all[df_all.columns[:-1]].to_numpy()
    Y = df_all['Label'].to_numpy()

    # 5折交叉验证
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # # 用svm分类器
    # # model = LogisticRegression()
    # model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)

    # # 用xgboost分类器
    # model = xgb.XGBClassifier(objective="multi:softmax", n_estimators=100, learning_rate=0.1, max_depth=6, random_state=0)




    # # 用随机森林分类器
    # model = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=0)

    
    train_data, test_data, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=42,stratify=Y)
    train_dataset = ADFS.CustomDataset(train_data, train_labels)
    test_dataset = ADFS.CustomDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    is_early_stopping = ADFS.EarlyStopping(num_trials=300,path='./trained_model/model.pth')




    INPUTS = X.shape[1]
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # model = ADFS.MLP(input_dim=INPUTS)
    model = ADFS.AFS_ZXM(input_dims=INPUTS,inputs_dim=inpus_dim)
    # model = ADFS.AdaFS_soft(input_dims=INPUTS,inputs_dim=inpus_dim,num=num )
    # model = ADFS.MvFS_MLP(input_dims=INPUTS,inputs_dim=inpus_dim,nums=num,num_selections=6)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-6)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,weight_decay=1e-6)

    # weight = torch.tensor([1,1,0.5,1,2]).cuda()

    # ct = utils.FocalLoss(weight, gamma=0.5)

    # 训练模型
    for epoch in range(300):
        model.train()
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(data.float())
            # loss = criterion(outputs, labels)
            loss = utils.loss_function(outputs, labels, loss_type='OE')
            # loss = ct(outputs, labels)
            loss.backward()
            optimizer.step()
        print('Epoch: %d, Loss: %.5f' % (epoch, loss.item()))
        model.eval()
        correct = 0
        total = 0
        for data, labels in test_loader:
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        # f1 = f1_score(labels.cpu().detach().numpy(), predicted.cpu().detach().numpy(),average='weighted')
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
        outputs = model(data.float())
        _, predicted = torch.max(outputs.data, 1)
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

    precision_score = precision_score(fin_targets, fin_outputs,average='weighted')
    recall_score = recall_score(fin_targets, fin_outputs,average='weighted')
    f1_score = f1_score(fin_targets, fin_outputs,average='weighted')
    print('Precision:', precision_score)
    print('Recall:', recall_score)
    print('F1:', f1_score)

    accuracy = 100 * correct / total
    print('Final accuracy: %.3f %%' % accuracy)


    

    
#     accs = []
#     prrcision = []
#     recall = []
#     f1s = []

#     GP = []
#     GR = []
#     FS = []
    

#     for train_index, test_index in skf.split(X, Y):
        
#         # fin_targets = []
#         # fin_outputs = []
#         X_train, X_test = X[train_index], X[test_index]
#         Y_train, Y_test = Y[train_index], Y[test_index]
#         model.fit(X_train, Y_train)
#         Y_pred = model.predict(X_test)
#         acc = accuracy_score(Y_test, Y_pred)
#         pr = precision_score(Y_test, Y_pred,average='weighted')
#         rc = recall_score(Y_test, Y_pred,average='weighted')
#         f1 = f1_score(Y_test, Y_pred,average='weighted')

#         GP = utils.gr_metrics(Y_pred, Y_test)[0]
#         GR = utils.gr_metrics(Y_pred, Y_test)[1]
#         FS = utils.gr_metrics(Y_pred, Y_test)[2]


#         accs.append(acc)
#         prrcision.append(pr)
#         recall.append(rc)
#         f1s.append(f1)

#     print('Accuracy:', np.mean(accs))
#     print('Precision:', np.mean(prrcision))
#     print('Recall:', np.mean(recall))
#     print('F1:', np.mean(f1s))

#     print('GP:', np.mean(GP))
#     print('GR:', np.mean(GR))
#     print('FS:', np.mean(FS))
# # 21
