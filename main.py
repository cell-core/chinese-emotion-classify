import pandas as pd
import chatbot
import data_tools


training_data=pd.read_csv("./data/trainC.csv")
test_data=pd.read_csv("./data/testC.csv")
# NNClassifier
#chatbot=chatbot.Chatbot(classifier='NNClassifier',epochs=70,batch_size=32,learning_rate=0.001)
#chatbot.train(training_data,"./model/nn_model.pt")
#chatbot=chatbot.Chatbot(classifier='NNClassifier',model_path="./model/nn_model.pt")

# LSTMClassifier
#chatbot=chatbot.Chatbot(classifier='LSTMClassifier',epochs=60,batch_size=32,learning_rate=0.001)
#chatbot.train(training_data,"./model/lstm_model.pt")
chatbot=chatbot.Chatbot(classifier='LSTMClassifier',model_path="./model/lstm_model.pt")

chatbot.evaluate(test_data)


# while True:
#     message=input("请输入消息：")
#     if(message == "退出" or message == "停止" or message == "stop"):
#         break
#     result=chatbot.predict(message)
#     print(result)