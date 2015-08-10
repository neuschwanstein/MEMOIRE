from selenium import webdriver
from selenium.webdriver.common.keys import Keys

stories = []

driver = webdriver.Chrome()
driver.get("http://finance.yahoo.com/news/provider-financial-times")
div = driver.find_element_by_id("mediatopstory_container")
lis = div.find_elements_by_tag_name("li")

for li in lis:
    story = {}
    story['title'] = li.find_element_by_class_name("txt").find_element_by_tag_name("a").text
    story['href'] = li.find_element_by_class_name("txt").find_element_by_tag_name("a").get_attribute("href")
    story['date'] = li.find_element_by_class_name("txt").find_element_by_tag_name("cite").text
    # story['story'] = li.find_element_by_class_name("preview").find_element_by_tag_name("p").text
    stories.append(story)

