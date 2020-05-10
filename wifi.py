import wifiCfg
from m5stack import *
from m5ui import *
from uiflow import *
from machine import UART,Pin
from math import cos, sin, pi
import urequests
import utime
def lcd_show(a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0, a8 = 0, a9 = 0):
    lcd.clear()
    color1 = lcd.RED
    color2 = lcd.BLACK
    colorB = lcd.MAROON

    lcd.font(lcd.FONT_Default)
    lcd.print('ESP32 + K210: WYL, SEU', 5, 220)

    lcd.font(lcd.FONT_DejaVu24)
    
    lcd.print('WATER METER', 70, 15)
    lcd.print('DETECTOR', 95, 45)
    
    lcd.print(str(a1), 20 + 20, 90)
    lcd.print(str(a2), 20 + 80, 90)
    lcd.print(str(a3), 20 + 140, 90)
    lcd.print(str(a4), 20 + 200, 90)
    lcd.print(str(a5), 20 + 260, 90)
    
    lcd.circle(60, 160, 30, 0xFFFFFF)
    lcd.line(60, 160,60 + int(30*sin(a6*36/180*pi)), 160-int(30*cos(a6*36/180*pi)), 0xFF0000)

    lcd.circle(60+70, 160, 30, 0xFFFFFF)
    lcd.line(60+70, 160,60+70 + int(30*sin(a7*36/180*pi)), 160-int(30*cos(a7*36/180*pi)), 0xFF0000)

    lcd.circle(60+140, 160, 30, 0xFFFFFF)
    lcd.line(60+140, 160,60+140 + int(30*sin(a8*36/180*pi)), 160-int(30*cos(a8*36/180*pi)), 0xFF0000)

    lcd.circle(60+210, 160, 30, 0xFFFFFF)
    lcd.line(60+210, 160,60+210 + int(30*sin(a9*36/180*pi)), 160-int(30*cos(a9*36/180*pi)), 0xFF0000)

    return 

uart = UART(1, baudrate=115200, rx=16,tx=17,timeout=10)
lcd.clear()
# auto connect wifi
wifiCfg.autoConnect(lcdShow=True)
lcd_show(a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0, a8 = 0, a9 = 0)

while True:

    if uart.any():
        bin_data = uart.readline()
        dat = '{}'.format(bin_data.decode())
        response = urequests.get('http://自己申请的域名网站/data/'+dat)
        response.text
        # lcd.print(dat,15,15)
        lcd_show(a1 = int(dat[0]), a2 = int(dat[1]), a3 = int(dat[2]), \
            a4 = int(dat[3]), a5 = int(dat[4]), \
            a6 = int(dat[9]), a7 = int(dat[8]), a8 = int(dat[7]), a9 = int(dat[6]))
