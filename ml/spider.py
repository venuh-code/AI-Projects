import requests
from lxml import etree
import re
import pymysql


class GovementSpider(object):
    def __init__(self):
        self.url = 'http://www.mca.gov.cn/article/sj/xzqh/2019/'
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        # 创建2个对象
        self.db = pymysql.connect(host='127.0.0.1', user='root', password='123456', database='mydb', charset='utf8')
        self.cursor = self.db.cursor()

    # 获取假链接
    def get_false_link(self):
        html = requests.get(url=self.url, headers=self.headers).text
        # 此处隐藏了真实的二级页面的url链接，真实的在假的响应网页中，通过js脚本生成，
        # 假的链接在网页中可以访问，但是爬取到的内容却不是我们想要的
        parse_html = etree.HTML(html)
        a_list = parse_html.xpath('//a[@class="artitlelist"]')
        for a in a_list:
            # get()方法:获取某个属性的值
            title = a.get('title')
            if title.endswith('代码'):
                # 获取到第1个就停止即可，第1个永远是最新的链接
                false_link = 'http://www.mca.gov.cn' + a.get('href')
                print("二级“假”链接的网址为", false_link)
                break
        # 提取真链接
        self.incr_spider(false_link)

    # 增量爬取函数
    def incr_spider(self, false_link):
        self.cursor.execute('select url from version where url=%s', [false_link])
        # fetchall: (('http://xxxx.html',),)
        result = self.cursor.fetchall()

        # not result:代表数据库version表中无数据
        if not result:
            self.get_true_link(false_link)
            # 可选操作: 数据库version表中只保留最新1条数据
            self.cursor.execute("delete from version")

            # 把爬取后的url插入到version表中
            self.cursor.execute('insert into version values(%s)', [false_link])
            self.db.commit()
        else:
            print('数据已是最新,无须爬取')

    # 获取真链接
    def get_true_link(self, false_link):
        # 先获取假链接的响应,然后根据响应获取真链接
        html = requests.get(url=false_link, headers=self.headers).text
        # 从二级页面的响应中提取真实的链接（此处为JS动态加载跳转的地址）
        re_bds = r'window.location.href="(.*?)"'
        pattern = re.compile(re_bds, re.S)
        true_link = pattern.findall(html)[0]

        self.save_data(true_link)  # 提取真链接的数据

    # 用xpath直接提取数据
    def save_data(self, true_link):
        html = requests.get(url=true_link, headers=self.headers).text

        # 基准xpath,提取每个信息的节点列表对象
        parse_html = etree.HTML(html)
        tr_list = parse_html.xpath('//tr[@height="19"]')
        for tr in tr_list:
            if tr.xpath('./td[2]/text()'):
                code = tr.xpath('./td[2]/text()')[0].strip()  # 行政区划代码
                name = tr.xpath('./td[3]/text()')[0].strip()  # 单位名称

            print(name, code)

    # 主函数
    def main(self):
        self.get_false_link()


if __name__ == '__main__':
    spider = GovementSpider()
    spider.main()