import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit


class DataCleaner():
    def __init__(self,dst_path="clean_data.csv"):
        self.dst_path=dst_path

    def drop_na(self,dataset):
        dataset.dropna(inplace=True)


class DataSplitter():
    def __init__(self,train_path="train.csv",test_path="test.csv"):
        self.train_path=train_path
        self.test_path=test_path

    def stratified_split(self,dataset,train_size=0.8,train_rows=None):
        # 将数据集按标签分为正样本和负样本
        positive_samples=dataset[dataset['label']==1]
        negative_samples=dataset[dataset['label']==0]

        # 分别对正样本和负样本进行分层抽样
        split=StratifiedShuffleSplit(n_splits=1,test_size=1-train_size,random_state=42)
        positive_train_idx,positive_test_idx=next(split.split(positive_samples,positive_samples['label']))
        negative_train_idx,negative_test_idx=next(split.split(negative_samples,negative_samples['label']))

        # 合并正样本和负样本的训练集和测试集
        train_idx=pd.concat([positive_samples.iloc[positive_train_idx],negative_samples.iloc[negative_train_idx]])
        test_idx=pd.concat([positive_samples.iloc[positive_test_idx],negative_samples.iloc[negative_test_idx]])

        # 打乱训练集和测试集的顺序
        train_idx=train_idx.sample(frac=1).reset_index(drop=True)
        test_idx=test_idx.sample(frac=1).reset_index(drop=True)

        # 只取前train_rows行数据
        if train_rows is not None:
            train_idx=train_idx[:train_rows]
            test_idx=test_idx[:int(train_rows*((1-train_size)/train_size))]

        # 保存训练集和测试集
        train_idx.to_csv(self.train_path,index=False)
        test_idx.to_csv(self.test_path,index=False)