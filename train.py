import pandas as pd
import chatbot


training_data=pd.read_csv("./data/weatherData.csv")
test_data=pd.read_csv("./data/coverTest.csv")

# NNClassifier
# chatbot=chatbot.Chatbot(classifier='NNClassifier',epochs=150,batch_size=32,learning_rate=0.001)
# chatbot.train(training_data,"./model/nn_model.pt")

# LSTMClassifier
chatbot=chatbot.Chatbot(classifier='LSTMClassifier',epochs=30,batch_size=32,learning_rate=0.001)
chatbot.train(training_data,"./model/lstm_model.pt")#训练模型并保存到./model/lstm_model.pt

chatbot.evaluate(test_data)