#-*-coding: utf-8 -*-

import urllib
import urllib.request
from bs4 import BeautifulSoup

'''
使用beautifulsoup下载图片
1、使用urllib.request 下载到网页内容
2、使用beautifulsoup匹配到所有的图片地址
3、指定文件路径
4、调用urllib.request.urlretrieve 下载图片
'''
def getAllImageLink(url, num = 0):
    html = urllib.request.urlopen(url)
    content = html.read()
    html.close()
    html_soup = BeautifulSoup(content, 'lxml')
    liResult = html_soup.findAll('div',attrs={"class":"box picblock col3"})

    for li in liResult:
        imageEntityArray = li.findAll('img')
        for image in imageEntityArray:
            link = image.get('src2')
            if link:
                imageName = image.get('alt')
                filesavepath = '/data/img/pictures/%s.jpg' % num
                link = link.replace('_s', '')
                urllib.request.urlretrieve("https:"+link,filesavepath)
                print(link, num)
                num += 1
                if num >= 1500:
                    return num

    try:
        next = html_soup.find_all('a', attrs={"class": "nextpage"})[0].get("href")
        if next:
            link = 'http://sc.chinaz.com/tupian/' + next
            print(link)
            num = getAllImageLink(link, num)

    except Exception:
        print('----end----')

    return num


if __name__ == '__main__':
    num = 0
    list = ['https://sc.chinaz.com/tupian/shanshuifengjing.html',
            'https://sc.chinaz.com/tupian/xiandaijianzhutupian.html',
            'https://sc.chinaz.com/tupian/guaishitupian.html',
            'https://sc.chinaz.com/tupian/haianshatantupian.html'
            ]

    for html in list:
        num = getAllImageLink(html, num)





