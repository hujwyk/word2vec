import os
import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------
# 数据预处理
# ----------------------

def load_texts(folder_path):
    """读取所有小说文本"""
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='gb18030') as f:
                texts.append(f.read())
    return texts

# 加载自定义词典（如没有，可省略）
# jieba.load_userdict("custom_dict.txt")

# 读取所有小说并分词
with open("stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = set(f.read().splitlines())

texts = load_texts("dataset")
sentences = []
for text in texts:
    words = jieba.lcut(text.replace("\n", ""))  # 去换行符
    filtered_words = [word for word in words if word not in stopwords]
    sentences.append(filtered_words)

# 构建词汇表
min_count = 5
word_counts = Counter()
for sentence in tqdm(sentences, desc="统计词频"):
    word_counts.update(sentence)
vocab = [word for word, count in word_counts.items() if count >= min_count]
print(f"过滤后词汇表大小: {len(vocab)}")
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}



'''
# 定义 Word2Vec 模型（Skip-Gram）

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):  # 词汇表大小，词向量维度
        super(Word2Vec, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim) # 中心词嵌入
        self.out_embed = nn.Embedding(vocab_size, embedding_dim) # 上下文词嵌入
        
    def forward(self, target, context, neg_samples):
        # 正样本损失 最大化目标词（中心词）与上下文词的共现概率
        target_emb = self.in_embed(target)
        context_emb = self.out_embed(context)
        pos_loss = torch.log(torch.sigmoid(torch.sum(target_emb * context_emb, dim=1)))
        
        # 负样本损失 最小化目标词与负样本词的共现概率
        neg_emb = self.out_embed(neg_samples)
        neg_loss = torch.log(torch.sigmoid(-torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze()))
        
        return -(pos_loss + neg_loss.sum(dim=1)).mean()
#最大化目标词与上下文词的共现概率，同时最小化目标词与负样本词的共现概率，来学习词向量。


# 数据集与训练

class SkipGramDataset(Dataset):
    def __init__(self, sentences, word_to_idx, window_size=3, neg_samples=3): #负采样数量3，每个目标词随机采样5个负样本
        print("正在初始化数据集，生成训练样本...")  # 添加这行
        self.data = []
        self.word_to_idx = word_to_idx
        self.idx_to_word = {v: k for k, v in word_to_idx.items()}
        self.neg_samples = neg_samples
        self.word_freq = torch.tensor([word_counts[word] for word in vocab], dtype=torch.float32)
        self.word_dist = self.word_freq.pow(0.75) / self.word_freq.pow(0.75).sum()
        
        # 生成训练对
        for sentence in tqdm(sentences, desc="生成训练样本"):
            sent = [word_to_idx[word] for word in sentence if word in word_to_idx]
            for i in range(len(sent)):
                target = sent[i]
                context_start = max(0, i - window_size)
                context_end = min(len(sent), i + window_size + 1)
                context = sent[context_start:i] + sent[i+1:context_end]
                for j in context:
                    self.data.append((target, j))
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target, context = self.data[idx]
        neg_samples = torch.multinomial(self.word_dist, self.neg_samples, replacement=True)
        return torch.tensor(target), torch.tensor(context), neg_samples

# 超参数
EMBEDDING_DIM = 64
BATCH_SIZE = 2048
EPOCHS = 3

# 初始化模型和优化器
dataset = SkipGramDataset(sentences, word_to_idx)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = Word2Vec(len(vocab), EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_idx, (target, context, neg_samples) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}",mininterval=2)):
        optimizer.zero_grad()
        loss = model(target, context, neg_samples)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 500 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")
    print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(dataloader)}")

# 保存词向量
word_vectors = model.in_embed.weight.data.cpu().numpy()
np.save("word_vectors.npy", word_vectors)
'''


# 加载词向量和词汇表
word_vectors = np.load("word_vectors.npy")
vocab = list(word_to_idx.keys())



# 姓氏列表
with open("surnames.txt", "r", encoding="utf-8") as f:
    surnames = [line.strip() for line in f]

import jieba.posseg as pseg
def is_person_pos(word):
    # 优化后的词性标注判断
    words = list(pseg.cut(word))
    # 组合词处理（如"郭靖"可能被拆分为"郭"nr+"靖"nr）
    nr_count = sum(1 for w, flag in words if flag == 'nr')
    return nr_count >= 1 and len(words) <= 2  # 允许单字姓或双字组合
def is_person_v2(word):
    score = 0
    # 规则0：长度过滤
    if is_person_pos(word):
        score += 2
    if word[0] in surnames:  # 姓氏过滤
        score += 1
    # 规则3：长度过滤
    if 2 <= len(word) <=4:
        score += 0.5
    else:
        score -= 1  # 长度不符合扣分

    return score >= 3


# 执行抽取并保存结果

# 抽取人名
person_names = []
person_counts = Counter()
for word in vocab:
    if 2 <= len(word) <=4 and is_person_v2(word):
        person_names.append(word)
        person_counts[word] = word_counts[word]  # 记录词频
# 筛选出现≥2次的人名
filtered_names = [name for name in person_names if person_counts[name] >= 2]


# 保存结果
pd.DataFrame({"人名": person_names}).to_csv("person_names.csv", index=False)


print(f"抽取到人名 {len(person_names)} 个")


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体


# 1. 降维到2D和3D
pca_2d = PCA(n_components=2)
word_vecs_2d = pca_2d.fit_transform(word_vectors)

pca_3d = PCA(n_components=3)
word_vecs_3d = pca_3d.fit_transform(word_vectors)

# 2. 可视化函数
def plot_word_vectors(words, vectors, counts, title, dim=2, freq_threshold=20):
    plt.figure(figsize=(12, 8))
    
    if dim == 2:
        plt.scatter(vectors[:, 0], vectors[:, 1], alpha=0.8)
        for i, word in enumerate(words):
            if counts[word] >= freq_threshold:
                plt.annotate(word,(vectors[i, 0], vectors[i, 1]), fontsize=8)
    else:
        ax = plt.axes(projection='3d')
        ax.scatter3D(vectors[:, 0], vectors[:, 1], vectors[:, 2], alpha=0.8)
        for i, word in enumerate(words):
            if counts[word] >= freq_threshold:
                ax.text(vectors[i, 0], vectors[i, 1], vectors[i, 2], word, fontsize=8)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

# 3. 可视化人名词向量
person_indices = [word_to_idx[name] for name in filtered_names if name in word_to_idx]
person_vecs_2d = word_vecs_2d[person_indices]
person_vecs_3d = word_vecs_3d[person_indices]

# 2D可视化
plot_word_vectors(filtered_names, person_vecs_2d, person_counts, "人名词向量分布 (2D)", dim=2, freq_threshold=5)

# 3D可视化
plot_word_vectors(filtered_names, person_vecs_3d, person_counts, "人名词向量分布 (3D)", dim=3, freq_threshold=5)

# 4. 保存降维后的向量
np.save("word_vectors_2d.npy", word_vecs_2d)
np.save("word_vectors_3d.npy", word_vecs_3d)
