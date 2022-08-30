import requests
from bs4 import BeautifulSoup
import re
import scrapy
import time
import base64
from selenium import webdriver

#important
html=requests.get('http://s.lvmama.com/ticket/K330300?keyword=%E9%98%B3%E6%B1%9F&k=0#list')
html.encoding='utf-8'

soup=BeautifulSoup(html.text,'lxml')

#print(soup.prettify()) #展示网页源代码
#print(soup.a) #获取a标签
#print(soup.a.attrs) #获取a标签的属性
#print(soup.body.div)
# data = soup.select('body > div.everything > div.main.clearfix > div.search-filter > div.search-body > div.search-lists > div:nth-child(1) > ul')
# for item in data:  # item.get_text()
#     print(item.get_text())



# theme =soup.find_all("d1",'product-details clearfix')
# for __theme in theme:
#    print(__theme)
# section = soup.find_all("div","product-section")
# for item in section:
#    name =item.contents[1].text
#    address =item.contents[5].text
#    print('name:'+name)
#    print('address:'+address)
#    print(item.contents)



product = soup.find_all("div","product-regular clearfix")
for item in product:
   try:
      print('scenicName:'+item.contents[3].a['title'])
      print('img:'+item.contents[3].img['src'])
      print('city:'+item.contents[9].span.text)
      print('price:'+item.contents[15].div.em.text)
      print('scenicScore:'+item.contents[15].ul.li.b.text)
      print('commenthref:'+item.contents[15].ul.li.next_sibling)
      address=item.contents[9].dd.text
      pattern = re.compile(r'\s+')
      address =re.sub(pattern,'',address)
      print('scenicAddress:'+address)
      print('\n\n')
   except:
      print("")

product2= soup.find_all("div","product-ticket-dropdown")
for item in product2:
   print(item.contents[2])


theme =soup.find_all('dl',"product-details clearfix")
for __theme in theme:
   massage=__theme.dd.text
   pattern = re.compile(r'\s+')
   massage = re.sub(pattern, '', massage)
   print(massage)
