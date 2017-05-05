import asyncio
import time
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import collections

urls = ['http://www.daishin.co.kr/ctx_kr/sc_stock/sg_stock_info/svc_kosdaq_total/KosdaqKsSise.shtml']


async def url_request(url):
    async with ClientSession() as session:  # 6
        async with session.get(url) as response:  # 7
            html = await response.read()  # 8
            bs = BeautifulSoup(html, 'lxml')  # 9
            raw_A = bs.find_all('a')
            return raw_A


async def get_html():  # 2
    htmls = [url_request(url) for url in urls]  # 3
    for i, completed in enumerate(asyncio.as_completed(htmls)):  # 4
        raw = await completed  # 5
        return raw


start_ = time.time()
event_loop = asyncio.get_event_loop()
try:
    data = event_loop.run_until_complete(get_html())  # 1
    stock_dict = collections.defaultdict()
    stocks = []
    for i in data:
        try:
            codes = str(i.text).split(' ')
            code, name = codes
            stock_dict[code] = name
            stocks.append(code)
        except Exception as e:
            continue
    print(len(set(stocks)))
    print(len(stock_dict.keys()), len(stock_dict.values()))
except:
    event_loop.close()
print("async 총걸린시간 {}".format(time.time() - start_))  # 약 1~2초 환경에따라 다름
