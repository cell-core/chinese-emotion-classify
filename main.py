import pandas as pd
import chatbot
import data_tools
from weather import get_text_weather_date
from colorama import init, Fore, Back, Style
from entitiy import EntityExtractor


training_data=pd.read_csv("./data/weatherData.csv")
test_data=pd.read_csv("./data/coverTest.csv")

# training_data=pd.read_csv("./data/trainC.csv")
# test_data=pd.read_csv("./data/testC.csv")
# NNClassifier
# chatbot=chatbot.Chatbot(classifier='NNClassifier',epochs=150,batch_size=32,learning_rate=0.001)
# chatbot.train(training_data,"./model/nn_model.pt")
#chatbot=chatbot.Chatbot(classifier='NNClassifier',model_path="./model/nn_model.pt")

# LSTMClassifier
# chatbot=chatbot.Chatbot(classifier='LSTMClassifier',epochs=30,batch_size=32,learning_rate=0.001)
# chatbot.train(training_data,"./model/lstm_model.pt")
chatbot=chatbot.Chatbot(classifier='LSTMClassifier',model_path="./model/lstm_model.pt")

#chatbot.evaluate(test_data)


# 初始化 colorama
init()
extractor = EntityExtractor()
weather_dict = {
    'time': None,
    'place': None,
    'date': None
}

print(Fore.YELLOW+'你好，我是Silly，一个专注天气预报的对话机器人'+Style.RESET_ALL)
while True:
    message=input("请输入消息：")
    if(message == "退出" or message == "停止" or message == "stop"):
        break
    result=chatbot.predict(message)
    print(result)
    if result[0][1] < 0.7:
        print(Fore.YELLOW+'你好，我能帮你查找中国大陆地区城市级别的当下和未来两天的天气情况'+Style.RESET_ALL)
    else:
        weather_dict = extractor.extract(message)
        print(weather_dict)
        if weather_dict['place'] is not None and weather_dict['date'] is not None:
            weather_data = get_text_weather_date(weather_dict['place'], weather_dict['date'], weather_dict['time'])
            print(Fore.YELLOW+weather_data+Style.RESET_ALL)
        elif weather_dict['place'] is None:
            print(Fore.YELLOW+'想查询哪里的天气呢？'+Style.RESET_ALL)
        elif weather_dict['time'] is None:
            print(Fore.YELLOW+'想查询什么时候的天气呢？'+Style.RESET_ALL)