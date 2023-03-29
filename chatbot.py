import jieba
import torch
import transformers
from transformers import BertTokenizer, BertModel
import re
import pandas as pd
from tqdm import tqdm

class JiebaTokenizer:
    def __init__(self):
        self.tokenizer=jieba

    def tokenize(self,message):
        # 使用jieba分词器对message进行分词，过滤掉空格和换行符等无用符号
        tokens=self.tokenizer.cut(message)
        return [token for token in tokens if token not in [' ','\n','\t']]

class LanguageModelFeaturizer:
    def __init__(self,model_name,model_weights):
        # 加载bert模型和tokenizer
        self.tokenizer=BertTokenizer.from_pretrained(model_name)
        self.model=BertModel.from_pretrained(model_weights)

    def featurize(self,message):
        # 对message使用tokenizer编码
        input_ids=torch.tensor([self.tokenizer.encode(message,add_special_tokens=True,max_length=512)])
        with torch.no_grad():
            # 获取编码后的文本特征
            features=self.model(input_ids)[1]
            return features.tolist()[0]

class NNClassifier:
    def __init__(self):
        self.model=None

    def train(self,X,y,epochs,batch_size,learning_rate):
        # 获取特征向量和标签的长度
        input_size=len(X[0])
        output_size=len(y[0])
        # 定义一个两层的神经网络
        self.model=torch.nn.Sequential(
            torch.nn.Linear(input_size,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,output_size),
            torch.nn.Sigmoid()
        )
        # 定义损失函数和优化器
        loss_fn=torch.nn.BCELoss()
        optimizer=torch.optim.Adam(self.model.parameters(),lr=learning_rate)
        # 开始训练
        for epoch in tqdm(range(epochs), desc="training"):
            for i in range(0,len(X),batch_size):
                X_batch=X[i:i+batch_size]
                y_batch=y[i:i+batch_size]
                y_pred=self.model(X_batch)
                loss=loss_fn(y_pred, y_batch.float())  # convert target tensor to float tensor
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # 更新进度条
            tqdm.write("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, epochs, loss.item()))

    def predict(self,X):
        # 预测意图
        y_pred=self.model(X)
        # 将预测结果的每个元素转化为浮点型
        return [float(y) for y in y_pred]

# 定义对话机器人类
class Chatbot():
    def __init__(self):
        self.tokenizer=JiebaTokenizer()
        self.featurizer=LanguageModelFeaturizer('bert-base-chinese','bert-base-chinese')
        self.classifier=NNClassifier()

    def train(self,training_data,epochs=100,batch_size=32,learning_rate=0.001):
        # 将训练数据转化为特征向量
        X=[]
        y=[]
        for row in tqdm(range(training_data.shape[0]), desc="featrurizing"):
            message=training_data.iloc[row]['review']
            features=self.featurizer.featurize(message)
            X.append(features)
            label=training_data.iloc[row]['label']
            y.append([label])
        # 训练分类器
        self.classifier.train(torch.tensor(X),torch.tensor(y),epochs,batch_size,learning_rate)

    def predict(self,message):
        # 提取特征向量
        features=self.featurizer.featurize(message)
        # 预测意图
        intent_probs=self.classifier.predict(torch.tensor([features]))
        return intent_probs

    def evaluate(self, test_data):
        # 将测试数据转化为特征向量
        X_test = []
        y_test = []
        for row in tqdm(range(test_data.shape[0]), desc="featrurizing_test"):
            message = test_data.iloc[row]['review']
            features = self.featurizer.featurize(message)
            X_test.append(features)
            label = test_data.iloc[row]['label']
            y_test.append(label) #注意与train()函数中y.append([label])不同

        # 将特征向量输入模型进行预测
        y_pred = self.classifier.predict(torch.tensor(X_test))
        y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred]

        # 计算各项预测参数
        accuracy = sum([1 if pred == true_label else 0 for pred, true_label in zip(y_pred, y_test)]) / len(y_test)
        precision = sum([1 if pred == 1 and true_label == 1 else 0 for pred, true_label in zip(y_pred, y_test)]) / sum(y_pred)
        recall = sum([1 if pred == 1 and true_label == 1 else 0 for pred, true_label in zip(y_pred, y_test)]) / sum(y_test)
        f1_score = 2 * precision * recall / (precision + recall)

        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1 Score: {:.4f}".format(f1_score))
