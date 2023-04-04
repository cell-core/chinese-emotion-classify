import pandas as pd
import chatbot
import data_tools

# df=pd.read_csv("database.csv")
# dc=data_tools.DataCleaner()
# dc.drop_na(df)
# #df.dropna(inplace=True)
# null_rows = df[df.isna().any(axis=1)].index
# print(null_rows)


# df=pd.read_csv("./data/database.csv")
# cleaner=data_tools.DataCleaner()
# cleaner.drop_na(df)
# spliter=data_tools.DataSplitter()
# spliter.stratified_split(df,train_rows=200)


training_data=pd.read_csv("./data/trainC.csv")
test_data=pd.read_csv("./data/testC.csv")
#chatbot=chatbot.Chatbot(classifier='NNClassifier',epochs=100,batch_size=32,learning_rate=0.001)
chatbot=chatbot.Chatbot(classifier='LSTMClassifier',epochs=100,batch_size=32,learning_rate=0.001)
chatbot.train(training_data)
chatbot.evaluate(test_data)

# while True:
#     message=input("请输入消息：")
#     if(message == "退出" or message == "停止" or message == "stop"):
#         break
#     result=chatbot.predict(message)
#     print(result)