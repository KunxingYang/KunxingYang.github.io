# 使用Python爬取前程无忧的谁看过我的简历
基本思路就是，使用get方法，获取页面http://i.51job.com/userset/resume_browsed.php?lang=c 的信息，然后使用beautiful对页面进行解析，提取需要的数据即可。其中要注意的是，访问上面的链接需要在登录的前提下，或者是在使用get时提交你的cookies，在登录51job的时候，使用chrome的分析工具查看请求头里的cookies信息，将其复制下来，在请求页面的时候将其填入就可以了。代码如下
```Python
# -*- coding:utf-8 -*-

import requests
from bs4 import BeautifulSoup
import datetime
import pandas as pd
import threading


cookies = {
        'cookies': "你的cookies"
    }


def check_update(last_times):
    url = 'https://login.51job.com/login.php?lang=c'
    headers = {
        'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    }

    session = requests.session()

    response = session.get('http://i.51job.com/userset/resume_browsed.php?lang=c', headers=headers, verify=False, cookies=cookies)

    soup = BeautifulSoup(response.content, "html.parser")
    times = soup.find("span", {"class": "c_orange"})
    # print(times.string)

    company = soup.find("div", {"class": "e qy"})
    # print(company.a['title'], company.a['href'])

    date = None
    time = None
    for i, child in enumerate(company.label.children):
        if i == 1:
            date = child.string
        elif i == 3:
            time = child.string

    if last_times == int(times.string):
        return [None, None, None, None, None]

    # 次数， 公司名称，公司无忧网址，日期，时间
    return [times.string, company.a['title'], company.a['href'], date, time]


def progress():
    # 读取之前存储的最后一条信息
    job = pd.read_excel('job.xlsx')
    # print(job.tail(1)['序号'].values)
    if not job.tail(1)['序号'].values:
        last_times = None
    else:
        last_times = int(job.tail(1)['序号'].values[0])
    # print(last_times)
    # 次数， 公司名称，公司无忧网址，日期，时间
    times, company_name, network_address, date, time = check_update(last_times)
    if times:
        new_df = pd.DataFrame([[times, company_name, network_address, date, time]],
                              columns=["序号", "公司名", "网址", "日期", "时间"])
        job = job.append(new_df)
        job.to_excel('job.xlsx')
        print("更新查看简历信息")
        print(times, ' ', company_name, ' ', network_address, ' ', date, ' ', time)

    # 重新设置定时器
    global timer
    timer = threading.Timer(1800.0, progress, [])
    timer.start()

if __name__ == "__main__":
    progress()
    timer = threading.Timer(1800.0, progress, [])
    timer.start()

```
