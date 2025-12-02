from selenium import webdriver
from selenium.webdriver.common.by import By
import pyautogui
import pyperclip
import time
from PIL import Image
import os, sys
import csv


""" PATH = "/Users/nick/chromedriver"
driver = webdriver.Chrome(PATH)

driver.get("https://www.mapcrunch.com/")
driver.fullscreen_window()

button = driver.find_element(By.ID, "go-button")
current_address = driver.find_element(By.ID, "address").text
button.click()

google_address = driver.find_element(By.XPATH, "//div[@class='gm-iv-address-link']/a").get_attribute('href')
 """
pyautogui.PAUSE = 1

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def share():
    pyautogui.moveTo(370, 167) #move to share button
    pyautogui.click()

def getShare():
    pyautogui.moveTo(170, 308) #move to share link
    pyautogui.click()
    pyautogui.rightClick()
    pyautogui.moveTo(270, 354) #move to right click + copy button
    pyautogui.click()

def go():
    pyautogui.moveTo(456, 164) #move to go button
    pyautogui.click()

def screenshot():
    im = pyautogui.screenshot(region=(0,388, 2880, 1202))
    return im.convert('RGB')

def take_screenshots(lat, long, x, y, z):
    starting_index = max([eval(i) for i in get_immediate_subdirectories('./imgs/')])+1 #gets list of all sub dirs, converts str->int, gets max
    save_dir = "./imgs/" + str(starting_index) + "/"                                           
    try:
        os.mkdir(save_dir)
    except:
        print(save_dir + " dir already exists")
    sc = screenshot()
    sc.save(save_dir + "0.jpg")
    for i in range(3):
        x += 90
        url = 'http://www.mapcrunch.com/p/'+ str(lat) + '_' + str(long) + '_' + str(x) + '_' + str(y) + '_' + str(z)
        pyautogui.moveTo(223, 61) #move to address bar
        pyautogui.click()
        pyautogui.write(url)
        pyautogui.press('enter')
        time.sleep(2)
        sc = screenshot()
        sc.save(save_dir + str(i+1) + ".jpg")

def main():
    time.sleep(2)
    print(pyautogui.size())
    file = open('latLong.csv', 'a')
    writer = csv.writer(file)
    share()
    getShare()
    share()
    go()
    copied = pyperclip.paste() # get init address

    for i in range(300):
        share()
        getShare()
        share()
        if copied == pyperclip.paste(): #check if copied, if not exit
            print(i)
            sys.exit("did not copy . . . quitting")

        address = pyperclip.paste()
        truncated_start_address = address.split('/')[-1]
        address_info_str = truncated_start_address.split('_')
        address_info = [eval(i) for i in address_info_str] # convert str array to int array
        writer.writerow([address_info[0], address_info[1]])
        time.sleep(1)
        take_screenshots(
            round(address_info[0], 5),
            round(address_info[1], 5), 
            round(address_info[2], 5),
            round(address_info[3], 5),
            round(address_info[4], 5))
        go()
        
    file.close()
if __name__ == "__main__":
    main()
