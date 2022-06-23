# -*- coding: utf-8 -*-
"""
Created on Jan 31 22:27:37 2021

@author: Evan
"""

# %%
import requests; from bs4 import BeautifulSoup;
import numpy as np; import pandas as pd
import csv
import datetime as dt
import sys
import sqlite3

# %%
#csv.field_size_limit(sys.maxsize)

def dynamic_data_entry(date, heading, link, text):
    c.execute("INSERT INTO SENS (datestamp, heading, link, content) VALUES (?,?,?,?)",
              (date, heading, link, text))
    conn.commit()

# %%
url = 'http://www.sharenet.co.za/v3/sens.php'

storytime = []
headline = []
links = []
output = pd.DataFrame(columns = ('link', 'time', 'headline'))
# %%
#Download headings, links and datetime for each heading

data = requests.get(url)
soup = BeautifulSoup(data.text, 'lxml')
table = soup.find(class_='table table-sm sens-table table-bordered')

# #Open SQL Cursor
# conn = sqlite3.connect('SENS.db')
# c = conn.cursor()
output = pd.DataFrame(columns = ('link', 'time', 'headline','text'))
rows = table.find_all('tr')
#if len(rows) == 2: continue
for row in rows[1:]:
    col = row.find_all('td')
    column_1 = col[0].string.strip()
    try:
        column_2 = col[2].find(class_="sens-story-link").string.strip()
    except:
        print(column_1 + str('Error: no headline'))
    uniqueID = col[2].find(class_="sens-story-link").get('href')
    storytime.append(dt.datetime.strptime(column_1, '%H:%M - %d %b %Y'))
    headline.append(column_2)
    uniquelink = 'http://www.sharenet.co.za/'+uniqueID
    links.append(uniquelink)
    r = requests.get(uniquelink)
    soup1=BeautifulSoup(r.text, 'lxml')
    textblock = soup1.find(class_ ='col pre-contain')
    heading = soup1.find(class_ = 'below-ad').text
    text = textblock.get_text(strip=True).replace("Wrap Text","")
    # text = text.encode('utf-8').strip()
    output = output.append(pd.DataFrame([[uniquelink, dt.datetime.strptime(column_1, '%H:%M - %d %b %Y'), column_2, text]], columns = ('link', 'time', 'headline','text')))
    # conn.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
    # dynamic_data_entry(dt.datetime.strptime(column_1, '%H:%M - %d %b %Y'), column_2, uniquelink, text)

# conn.close()
