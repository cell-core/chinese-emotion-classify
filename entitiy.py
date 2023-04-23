import datetime
import re
from typing import Optional
from dateutil.parser import parse
from pyhanlp import *


def text_to_date(text_date: str) -> Optional[datetime.date]:
    """convert text based Chinese date info into datatime object

    if the convert is not supprted will return None
    """

    today = datetime.datetime.now()
    one_more_day = datetime.timedelta(days=1)

    if text_date == "今天":
        return today.date()
    if text_date == "明天":
        return (today + one_more_day).date()
    if text_date == "后天":
        return (today + one_more_day * 2).date()

    # Not supported by weather API provider freely
    if text_date == "大后天":
        # return 3
        return (today + one_more_day * 3).date()

    # follow APIs are not supported by weather API provider freely
    if text_date == "昨天":
        return None
    if text_date == "前天":
        return None
    if text_date == "大前天":
        return None


class EntityExtractor:
    def __init__(self):
        #self.segment = HanLP.newSegment().enablePlaceRecognize(True).enableTimeRecognize(True)
        self.segment = HanLP.newSegment().enableAllNamedEntityRecognize(True)
        self.text = None
        self.entities  = {
            'time': None,
            'place': None,
            'date': None
        }

    def hanLP_extractor(self):
        term_list = self.segment.seg(self.text)
        for term in term_list:
            if str(term.nature) == 't':
                self.entities['time'] = str(term.word)
                self.entities['date'] = text_to_date(self.entities['time'])
            elif str(term.nature) == 'ns':
                self.entities['place'] = str(term.word)

    def parse_extractor(self):
        try:
            date = parse(self.text, fuzzy=True)
            self.entities['time'] = date#bug: 4.22北京--->找到日期：2023年04月04日
            self.entities['date'] = date.date()
            print(f'找到日期：{date.strftime("%Y年%m月%d日")}')
        except ValueError:
            pass

    def re_extractor(self):
        pattern = r"(\d{2})([年/.\-])(\d{1,2})([月/.\-])(\d{1,2})|(\d{1,2})([月/.\-])(\d{1,2})|(\d{1,2})([日号])"
        date_pattern = re.compile(pattern)
        match = date_pattern.search(self.text)
        if match:
            today = datetime.date.today()
            year = today.year
            month = today.month
            day = today.day
            if match.group(1):
                year = '20' + match.group(1)
                month = match.group(3)
                day = match.group(5)
            elif match.group(6):
                month = match.group(6)
                day = match.group(8)
            else:
                day = match.group(9)
            date_str = f"{year}年{month}月{day}日"
            date_obj = datetime.datetime.strptime(date_str, "%Y年%m月%d日")
            delta = date_obj.date() - today
            diff_days = delta.days
            if diff_days >=0 and diff_days <= 2:
                self.entities['time'] = f"{year}年{month}月{day}日"
                self.entities['date'] = date_obj.date()
            else:
                print("仅支持查询今天到后天的天气")#------------to develop

    def lookup_table(self):
        date_pattern = re.compile(r"[今|明|后]天")
        match = date_pattern.findall(self.text)
        if match:
            self.entities['time'] = match[0]
            self.entities['date'] = text_to_date(self.entities['time'])

    def extract(self,text):
        self.text=text
        self.hanLP_extractor()
        #self.parse_extractor()
        self.re_extractor()
        self.lookup_table()#会覆盖hanLP_extractor中的[今|明|后]天
        return self.entities