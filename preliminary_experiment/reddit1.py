
import ast
import torch
import torch.nn as nn
import string
import nltk # 用于处理人类语言数据的符号和统计自然语言处理（NLP）。它提供了文本处理库用于分类、标记、语法分析、语义推理等。
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.stem.wordnet import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import model_adfs as ADFS
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from tools import utils


reddit = utils.load_df('reddit_500')



# 定义一个 Min-Max 归一化的函数
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
    num_pos_token,  num_pos_sent = [], []
    for str in data['Post']:
        temp_token, temp_sent = 0, 0
        for pos in string_list(str):
          # 单词数
          temp_token += len(nltk.word_tokenize(pos))
          # 句子数
          temp_sent += len(nltk.tokenize.sent_tokenize(pos, language='english'))
        num_pos_token.append(temp_token)
        num_pos_sent.append(temp_sent)
    features = {'pos_token': num_pos_token,
                'pos_sent': num_pos_sent, }
    
    df =  pd.DataFrame(features, columns=['pos_token', 'pos_sent'])
    # 对 DataFrame 的每列应用归一化
    normalized_df = df.apply(log_normalize)
    # normalized_df = df.apply(min_max_normalize)
    # normalized_df.to_csv('features.csv', index=False)
    # df.to_csv('features.csv', index=False)

    return normalized_df
    # return df
    

# 得到词性标定（POS）的所有标记
# todo 
def get_all_tags(data):
    print("Processing POS features ...")
    tags_all = []
    for tags in data['Post']:
        for body in string_list(tags):
          tagged_text = nltk.pos_tag(nltk.word_tokenize(body))
          for word, tag in tagged_text:
              if tag not in tags_all:
                  tags_all.append(tag)
    # 得到的tags_all是一个列表，里面包含了所有的词性标记
    # print(len(tags_all))
    # print(tags_all)
    return tags_all


# get_all_tags(reddit)
# 提供的数据集中提取词性标记（POS）的统计特征。
def f_pos(data, tags_all):
  # tag_count: 存储每个标题中各词性标记的出现次数。
  # tag_count_body: 存储每个正文中各词性标记的出现次数
  tag_dict, tag_count_body = {}, {}
  # print(len(tags_all))
  for tag in tqdm(tags_all):
      tag_dict[tag] = 0
      tag_count_body[tag] = []
  for posts in tqdm(data['Post']):
      for tag in tags_all:
          tag_dict[tag] = 0
      for text_sentence in string_list(posts):
        tagged_text = nltk.pos_tag(nltk.word_tokenize(text_sentence))
        for word, tag in tagged_text:
            tag_dict[tag] += 1
          # print(tag_dict.values(), tag_dict.keys(),105)
      for count, tag in zip(tag_dict.values(), tag_dict.keys()):
          # 将每个用户的pos向量进行记录
          tag_count_body[tag].append(count)

  pd.DataFrame(tag_count_body, index=None).to_csv('tag_count_body.csv', index=False)
  df = pd.DataFrame(tag_count_body, index=None)
  log_df = df.apply(log_normalize)
#   log_df = df.apply(min_max_normalize)
  # log_df.to_csv('log_tag_count_body.csv', index=False)
  return log_df
#   return df


def get_ifidf_features(data):
    print("Processing TF-IDF features ...")
    X = []
    for t in data['Post']:
        X.append(t)
    # 这个对象用于将文本转换为词频矩阵。它配置了三个参数：移除英文停用词，只考虑单个词的n-grams，最多考虑频率最高的50个词。
    count_vect = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_features=50)
    X_counts = count_vect.fit_transform(X)
    word = count_vect.get_feature_names_out()
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    # 将稀疏矩阵转换为稠密矩阵，并创建DataFrame
    df_tfidf = pd.DataFrame(X_tfidf.todense(), columns=word)
    # 保存DataFrame到CSV文件
    # df_tfidf.to_csv('tfidf_features.csv', index=False)
    # return pd.DataFrame(X_tfidf.todense(), columns=word)
    return df_tfidf


def get_lda_features(data, topic_num=10):
    print("Processing Topics features ...")
    # 定义一个名为cleaning的内部函数，用于对文章进行预处理
    def cleaning(article):
        # string.punctuation 是一个包含了所有标点符号的字符串，例如句号、逗号、叹号等
        punctuation = set(string.punctuation)
        # 词形归并是将单词转换为它们的基本形式（称为词元或词根）的过程，通常是将单词转换为它们的词汇原型。
        lemmatize = WordNetLemmatizer()
        # 将文章转换为小写，并移除停用词
        one = " ".join([i for i in article.lower().split() if i not in stopwords])
        # 移除标点符号
        two = "".join(i for i in one if i not in punctuation)
        # 对单词进行词形还原
        three = " ".join(lemmatize.lemmatize(i) for i in two.lower().split())
        return three
    
    
    # 对新文档进行预处理并转换为词袋表示
    def pred_new(doc):
        one = cleaning(doc).split()
        two = dictionary.doc2bow(one)
        return two

    # 从输入的数据集中加载标题和正文，将它们合并为单一的文本字符串列表。
    def load_title_body(data):
        text =[]
        for i in range(len(data["Label"])):
            # 去除文中的双引号？？？
            text.append(data["Post"][i])
        return text


    stopwords = set(nltk.corpus.stopwords.words('english'))

    # 多余的操作
    text_all = load_title_body(data)
    df = pd.DataFrame({'text': text_all}, index=None)
    # 对DataFrame中的每个文本应用预处理函数
    text = df.map(cleaning)['text']
    text_list = []
    for t in text:
        temp = t.split()
        # 去除停用词
        text_list.append([i for i in temp if i not in stopwords])

    # 创建词汇词典
    dictionary = corpora.Dictionary(text_list)

    # corpus = [dictionary.doc2bow(doc) for doc in text_list]

    # 通过词典将文本转换为词袋表示
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]
    ldamodel = LdaModel(doc_term_matrix, num_topics=topic_num, id2word = dictionary, passes=50)
    # ldamodel = LdaModel(corpus, num_topics=topic_num, id2word = dictionary, passes=50)

    # lda_display = gensimvis.prepare(ldamodel, corpus, dictionary, sort_topics=False)


    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary, sort_topics=False)
    # # pyLDAvis.display(vis)
    # pyLDAvis.save_html(vis, 'lda_visualization.html')

    # 展示主题模型

    probs = []

    for text in text_all:
        prob = ldamodel[(pred_new(text))]
        # 每种主题的可能性
        d = dict(prob)
        # 防止出现错误
        for i in range(topic_num):
            if i not in d.keys():
                d[i] = 0
        temp = []
        for i in range(topic_num):
            temp.append(d[i])
        probs.append(temp)
    
    # df =  pd.DataFrame(probs, index=None).to_csv('lda_features.csv', index=False)
    return pd.DataFrame(probs, index=None)    
    # return pd.DataFrame(probs, index=None)    
    # return df

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




class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = np.array(data)  # 将数据转换为 NumPy 数组
        self.labels = np.array(labels)  # 将标签转换为 NumPy 数组
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.long)




if __name__ == '__main__':
    seed=2024
    print(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    reddit = utils.load_df('reddit_500')
    num_features = 4
    # if num_features == 4:
    #     df_basic = get_satisc_features(reddit)
    #     df_tfidf = get_ifidf_features(reddit)
    #     tags_all = get_all_tags(reddit)
    #     df_pos= f_pos(reddit, tags_all)
    #     # print(pos_dims)
    #     df_topic = get_lda_features(reddit,topic_num=10)
    #     df_emotion = get_emotion_features(reddit)
    #     df_sd = get_sd_features(reddit)
    #     # 字段的顺序对模型的性能有很大的影响
    #     df_features = pd.concat([df_basic,df_pos,df_sd,df_tfidf,df_topic,df_emotion], axis=1)
    #     df_all = pd.concat([df_features, reddit['Label']], axis=1)
    #     df_all.to_csv('./reddit1_data/srd_feature.csv', index=False)
    # else:
    #     raise ValueError("Error: number of features groups")
    
    inputs_dim = {
        'st':2,
        'pos': 44,
        'sd': 4,
        'tfidf': 50,
        'topic': 10,
        'emotion': 4,

    }
    

    # df_all = pd.read_csv('./srd_features.csv')
    # df_all = pd.read_csv('./All_unnorn_srd_features.csv')
    df_all = pd.read_csv('./reddit1_data/srd_feature.csv')

    # print(df_features.shape)

    X = df_all[df_all.columns[:-1]].to_numpy()
    Y = df_all['Label'].to_numpy()

    # 划分数据集为训练集和测试集
    train_data, test_data, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=42,stratify=Y)
    # 创建训练集和测试集的Dataset实例
    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)
    # 创建 DataLoader 实例
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],}
    
    for i in range(5):
      is_early_stopping = ADFS.EarlyStopping(num_trials=300,path='./trianed_model/reddit1_model.pth')
      input_dim = X.shape[1]
      print(input_dim)
      output_dim = 5

      model = ADFS.AdaFS_soft(input_dim,inputs_dim,num=6)
      # model = model.AdaFS_hard(input_dim,inputs_dim)
      # model = ADFS.MLP(input_dim)
      model = model.cuda()
      # 定义损失函数和优化器
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5)

      # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=)
      # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
      print('Start Training...')

      for epoch in range(300):
          fin_targets = []
          fin_outputs = []
          model.train()
          for i, (data, labels) in enumerate(train_loader):
              data, labels = data.cuda(), labels.cuda()
              # model.zero_grad()
              optimizer.zero_grad()
              torch.autograd.set_detect_anomaly(True)
              outputs = model(data)
              loss = criterion(outputs, labels)
              _, predicted = torch.max(outputs.data, 1)
              # loss = utils.loss_function(outputs, labels,loss_type='OE')
              loss.backward()
              optimizer.step()
          model.eval()
          correct, total = 0, 0
          acc = 0
          with torch.no_grad():
              for data, labels in test_loader:
                  data, labels = data.cuda(), labels.cuda()
                  outputs = model(data)
                  _, predicted = torch.max(outputs.data, 1)
                  total += labels.size(0)
                  correct += (predicted == labels).sum().item()
                  acc = correct / total
                  fin_targets.append(labels.cpu().detach().numpy())
                  fin_outputs.append(predicted.cpu().detach().numpy())
              fin_outputs =  np.hstack(fin_outputs)
              fin_targets =  np.hstack(fin_targets)
              m = utils.gr_metrics(fin_outputs, fin_targets)
              if is_early_stopping.is_continuable(model, acc) == False:
                  print(f'validation: best auc: {is_early_stopping.best_accuracy}')
                  break

      print('Finished Training')

      # 加载最佳模型
      model.load_state_dict(torch.load(is_early_stopping.save_path)['state_dict'])
      model.eval()
      M = utils.test(model, test_loader, 'cuda')
      results['accuracy'].append(M[0])
      results['precision'].append(M[1])
      results['recall'].append(M[2])


# print(results)
    print(f"GP: {np.mean(results['accuracy'])}")
    print(f"GR: {np.mean(results['precision'])}")
    print(f"FS: {np.mean(results['recall'])}")

  

 


# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_name', default='reddit', help='criteo, avazu, movielens1M')
# parser.add_argument('--model_name', default='NoSlct', help='NoSlct, AdaFS_soft, AdaFS_hard')
# parser.add_argument('--k', type=int, default=0) #选取的特征数,for AdaFS_hard
# parser.add_argument('--useWeight', type=bool, default=True) 
# parser.add_argument('--reWeight', type=bool, default=True)
# parser.add_argument('--useBN', type=bool, default=True)
# parser.add_argument('--mlp_dims', type=int, default=[55,5], help='original=16')
# parser.add_argument('--embed_dim', type=int, default=110, help='original=16')
# parser.add_argument('--epoch', type=int, default=[2,300], nargs='+', help='pretrain/main_train epochs') 
# parser.add_argument('--learning_rate', type=float, default=0.001)
# parser.add_argument('--learning_rate_darts', type=float, default=0.0001)
# parser.add_argument('--batch_size', type=int, default=16)
# parser.add_argument('--darts_frequency', type=int, default=10)
# parser.add_argument('--weight_decay', type=float, default=1e-6)
# parser.add_argument('--dropout',type=int, default=0.2)
# parser.add_argument('--num', type=int, default=5)
# parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda:0')
# parser.add_argument('--save_dir', default='chkpt')
# parser.add_argument('--add_zero',default=False, help='Whether to add a useless feature')
# parser.add_argument('--controller',default=False, help='True:Use controller in model; False:Do not use controller')
# parser.add_argument('--pretrain',type=int, default=1, help='0:pretrain to converge, 1:pretrain, 2:no pretrain') 