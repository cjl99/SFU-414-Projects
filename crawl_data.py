from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import os
import time

count = 1024
for page in range(100):
    url  = "https://www.zerochan.net/?p=" + str(page+51)
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    items = soup.find_all('img')

    folder_path = './images/'
    if os.path.exists(folder_path) == False:  # 判断文件夹是否已经存在
        os.makedirs(folder_path)  # 创建文件夹

    for index,item in enumerate(items):
        if item:		
            html = requests.get(item.get('src'))   # get函数获取图片链接地址，requests发送访问请求
            count += 1
            img_name = folder_path + str(count) +'.png'
            with open(img_name, 'wb') as file:  # 以byte形式将图片数据写入
                file.write(html.content)
                file.flush()
            file.close()  # 关闭文件
            print('downlode image , page ', count, page)
            # time.sleep(1)  # 自定义延时
print('completed!')
