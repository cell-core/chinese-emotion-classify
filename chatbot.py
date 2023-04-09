import jieba
import torch
import transformers
from transformers import BertTokenizer, BertModel
import re
import pandas as pd
from tqdm import tqdm
import os


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

    def featurize(self,message,mode):
        # 对message使用tokenizer编码
        input_ids=torch.tensor([self.tokenizer.encode(message,add_special_tokens=True,max_length=100,pad_to_max_length=True,truncation=True)])
        with torch.no_grad():
            # 获取编码后的文本特征
            if(mode=='NNClassifier'):
                features=self.model(input_ids)[1]# 返回文本平均特征
                return features.tolist()[0]
            elif(mode=='LSTMClassifier'):
                features=self.model(input_ids)[0]# 返回文字序列特征
                return features.tolist()[0]#torch.Size([1, 177, 768])  torch.Size([1, 100, 768])


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
        num_batches = len(X) // batch_size
        for epoch in tqdm(range(epochs), desc="training"):
            running_loss = 0.0
            for i in range(0,len(X),batch_size):
                X_batch=X[i:i+batch_size]
                y_batch=y[i:i+batch_size]
                y_pred=self.model(X_batch)
                loss=loss_fn(y_pred, y_batch.float())  # convert target tensor to float tensor
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / num_batches
            # 更新进度条
            tqdm.write("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, epochs, epoch_loss))

    def predict(self,X):
        # 预测意图
        y_pred=self.model(X)
        # 将预测结果的每个元素转化为浮点型
        return [float(y) for y in y_pred]


class LSTMClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = None
        self.fc = None
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n.squeeze(0))#(num_layers * num_directions, batch_size, hidden_size)--->(batch_size, hidden_size)
        out = self.activation(out)
        return out

    def train(self, X, y, epochs, batch_size, learning_rate, hidden_size=120):
        # 获取特征向量和标签的长度
        input_size=len(X[0][0])#(batch_size, sequence_length, input_size)
        output_size=len(y[0])
        # 构造网络
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)# 把网络构造放在这里主要是因为需要确认输入输出大小，之后应该放入初始化方法中
        self.fc = torch.nn.Linear(hidden_size, output_size)
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = torch.nn.BCELoss()
        num_batches = len(X) // batch_size
        for epoch in tqdm(range(epochs), desc="training"):
            running_loss = 0.0
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                y_pred = self.forward(X_batch)
                loss = loss_fn(y_pred, y_batch.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / num_batches
            # 更新进度条
            tqdm.write("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, epochs, epoch_loss))

    def predict(self, X):
        input_tensor = torch.tensor(X, dtype=torch.float32)
        y_pred = self.forward(input_tensor).detach().numpy()
        return y_pred.ravel()


# 定义对话机器人类
class Chatbot():
    def __init__(self,classifier,epochs=100,batch_size=32,learning_rate=0.001,model_path=None):
        self.tokenizer=JiebaTokenizer()
        self.featurizer=LanguageModelFeaturizer('bert-base-chinese','bert-base-chinese')
        self.epochs=epochs
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.mode=classifier#用于决定特征提取方法
        if(classifier=='NNClassifier'):
            self.classifier=NNClassifier()
        elif(classifier=='LSTMClassifier'):
            self.classifier=LSTMClassifier()
        if model_path is not None and os.path.exists(model_path):
            self.classifier= torch.load(model_path)# 覆盖了上面的self.classifier赋值
            self.trained = True
            self.loaded = True
        else:
            self.trained = False
            self.loaded = False

    def train(self,training_data,model_path='./model/demo_model.pt'):
        # 将训练数据转化为特征向量
        X=[]
        y=[]
        for row in tqdm(range(training_data.shape[0]), desc="featrurizing"):
            message=training_data.iloc[row]['review']
            features=self.featurizer.featurize(message,self.mode)
            X.append(features)
            label=training_data.iloc[row]['label']
            y.append([label])
        # 训练分类器
        self.classifier.train(torch.tensor(X),torch.tensor(y),self.epochs,self.batch_size,self.learning_rate)
        # 保存模型
        if model_path is not None:
            torch.save(self.classifier, model_path)#实际保存classifier实例
        self.trained = True

    def predict(self,message):
        # 提取特征向量
        features=self.featurizer.featurize(message,self.mode)
        # 预测意图
        intent_probs=self.classifier.predict(torch.tensor([features]))
        return intent_probs

    def evaluate(self, test_data):
        # 若模型未训练，则不能进行预测和评估
        if not self.trained:
            print("Please train the model first!")
            return
        # 将测试数据转化为特征向量
        X_test = []
        y_test = []
        for row in tqdm(range(test_data.shape[0]), desc="featrurizing_test"):
            message = test_data.iloc[row]['review']
            features = self.featurizer.featurize(message,self.mode)
            X_test.append(features)
            label = test_data.iloc[row]['label']
            y_test.append(label) #注意与train()函数中y.append([label])不同

        # 将特征向量输入模型进行预测
        y_prob = self.classifier.predict(torch.tensor(X_test))
        y_pred = [1 if prob >= 0.5 else 0 for prob in y_prob]

        # 计算各项预测参数
        accuracy = sum([1 if pred == true_label else 0 for pred, true_label in zip(y_pred, y_test)]) / len(y_test)
        precision = sum([1 if pred == 1 and true_label == 1 else 0 for pred, true_label in zip(y_pred, y_test)]) / sum(y_pred)
        recall = sum([1 if pred == 1 and true_label == 1 else 0 for pred, true_label in zip(y_pred, y_test)]) / sum(y_test)
        f1_score = 2 * precision * recall / (precision + recall)

        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1 Score: {:.4f}".format(f1_score))

        # 保存分类错误的样本到文件中
        wrong_samples_to0 = []
        wrong_samples_to1 = [] # 被错误归类为1的样本
        prob_to0 = []
        prob_to1 = [] # 对应样本模型预测的prob值 
        for i in tqdm(range(len(y_pred)), desc="wrongData"):
            if y_pred[i] != y_test[i]:
                if y_pred[i] == 0:
                    wrong_samples_to0.append(test_data.iloc[i])
                    prob_to0.append(round(y_prob[i], 2))
                else:
                    wrong_samples_to1.append(test_data.iloc[i])
                    prob_to1.append(round(y_prob[i], 2))
        wrong_df0 = pd.DataFrame(wrong_samples_to0)
        wrong_df1 = pd.DataFrame(wrong_samples_to1)
        wrong_df0.insert(loc=1, column='prob', value=prob_to0)
        wrong_df1.insert(loc=1, column='prob', value=prob_to1)
        wrong_df0.to_csv('./data/wrong/wrongTo0.csv', index=False)
        wrong_df1.to_csv('./data/wrong/wrongTo1.csv', index=False)

        # 写入训练信息
        if not self.loaded:
            train_info=pd.read_csv('./data/train_info.csv')
            new_row = {'classifier': self.mode, 'epochs': self.epochs, 'batch_size': self.batch_size, 'learning_rate': self.learning_rate, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}
            train_info=train_info.append(new_row, ignore_index=True)
            train_info.to_csv('./data/train_info.csv', index=False)