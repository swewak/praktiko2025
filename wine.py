from selenium import webdriver
from bs4 import BeautifulSoup
import pandas

driver = webdriver.Chrome()

driver.get("https://ratcatcher.ru/media/summer_prac/parcing/4/1c31e19ca54a.html")
js_content = driver.page_source
soup = BeautifulSoup(js_content, "html.parser")
table = soup.find("table", {"id": "wines_SPA"})
columns_tags = table.find("thead")
columns_massive = columns_tags.find_all("th")
columnsmassive=[]
for column in columns_massive:
    txtcolumn=column.text
    columnsmassive.append(txtcolumn)
df = pandas.DataFrame(columns=columnsmassive)
df.to_csv('wine1.csv', index=False, encoding='cp1251')


driver.get("https://ratcatcher.ru/media/summer_prac/parcing/4/index.html")
js1_content = driver.page_source
soup1 = BeautifulSoup(js1_content, "html.parser")
links=soup1.find_all("a")
linksmassive=[]
for link in links:
    txtlink=link.text
    linksmassive.append(txtlink)
rows = []
for linkm in linksmassive:
    driver.get(linkm)
    js_content = driver.page_source
    soup = BeautifulSoup(js_content, "html.parser")
    table = soup.find("table", {"id": "wines_SPA"})
    columns_tags = table.find("thead")
    columns_massive = columns_tags.find_all("th")
    headers_table = [item.text for item in columns_massive]
    table_body_tag = table.find("tbody")
    rows_tags = table_body_tag.find_all("tr")
    print(rows_tags)
    for row_tag in rows_tags:
        rows_items_tags = row_tag.find_all("td")
        rows_items = [tag_row_item.text for tag_row_item in rows_items_tags]
        rows.append(rows_items)
df1 = pandas.DataFrame(columns=rows)
df1.to_csv('wine1.csv', mode='a', index=False, encoding='cp1251')
