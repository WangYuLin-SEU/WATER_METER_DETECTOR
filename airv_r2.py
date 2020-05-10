"""
这是一个仪表图像处理实例，基于k210芯片，歪朵拉开发板。
同级根目录下mnist.kmodel需放置于sd卡，请讲sd卡命名为sd。
TODO：
1、更好的针对印刷体数据集的数据增广；
2、指针自适应特征颜色提取范围；
"""
from fpioa_manager import fm, board_info
from machine import UART
import utime
fm.register(board_info.PIN9,fm.fpioa.UART2_TX)
fm.register(board_info.PIN10,fm.fpioa.UART2_RX)
uart_B = UART(UART.UART2, 115200, 8, None, 1, timeout=10)
import sensor, image, time, lcd, math
import KPU as kpu
#task = kpu.load("/sd/paste_mnist.kmodel")
task = kpu.load("/sd/mnist.kmodel")
info=kpu.netinfo(task)

lcd.init(freq=15000000)
sensor.reset()                      # Reset and initialize the sensor. It will
                                    # run automatically, call sensor.run(0) to stop
sensor.set_pixformat(sensor.RGB565) # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)   # Set frame size to QVGA (320x240)
sensor.set_vflip(True)
sensor.set_auto_gain(True)
sensor.set_auto_whitebal(True)
sensor.set_gainceiling(8)
sensor.skip_frames(time = 2000)     # Wait for settings take effect.
clock = time.clock()                # Create a clock object to track the FPS.

def mnist_run(img, dx, dy, dis, x00 =0, y00 = 80, nnn = 2):
    if nnn == 4:
        x00 = x00
        dy = dy
    img0 = img.copy((x00+dis*nnn,y00+nnn*0, dx, dy))
    #img0.mean(2, threshold=True, offset=1, invert=True)  #A
    img0.median(2, percentile=0.3, threshold=True, offset=-3, invert=True)
    #img0.midpoint(2, bias=0.3, threshold=True, offset=0, invert=True)
    #img0.mode(2, threshold=True, offset=0, invert=True)  #B

    #img0.binary([(110,255)], invert = True)
    for dx0 in range(dx):
        for dy0 in range(dy):
            a0 = img0.get_pixel(dx0,dy0)
            img.set_pixel(x00+dis*nnn+dx0,y00+nnn*0+dy0,a0)
    #img1 = img0.copy((1,1, dx-1, dy-1))
    img1 = img0
    img1 = img1.resize(28,28)
    img1 = img1.to_grayscale(1)
    img1.pix_to_ai()
    fmap=kpu.forward(task,img1)
    plist=fmap[:]
    pmax=max(plist)
    max_index=plist.index(pmax)
    kpu.fmap_free(fmap)
    return max_index, pmax

def search_col(x_input, y_input, img, width = 320, height = 240):
    x_l = []
    y_l = []
    for x in range(x_input - 32,x_input + 32):
        for y in range(y_input - 32,y_input + 32):
            if math.sqrt((x-x_input)*(x-x_input) + (y-y_input)*(y-y_input))<32 and math.sqrt((x-x_input)*(x-x_input) + (y-y_input)*(y-y_input))>14:
                col = img.get_pixel(x,y)
                if col[0]>120 and col[1]<100 and col[2]<100:
                    x_l.append(x-x_input)
                    y_l.append(-y+y_input)
                    #img.set_pixel(x,y,(255,255,255))
                #else:
                    #img.set_pixel(x,y,(0,0,0))
    angle_count = 0
    le = 0
    x_c = 0
    y_c = 0
    for i in range(len(x_l)):
        leng = math.sqrt(x_l[i]**2 + y_l[i]**2)
        le = le + leng
        angle_count = angle_count + math.acos(y_l[i]/leng)*leng
        x_c = x_c + x_l[i]
        y_c = y_c + y_l[i]
    if le == 0:
        angle = 0
    else:
        angle = angle_count/le
    dx = 0
    dy = 0
    dx = int(30 * math.sin(angle))
    dy = int(30 * math.cos(angle))
    if x_c < 0:
        angle = -angle + 2*math.pi
        dx = -dx
    img.draw_line((x_input, y_input,x_input+dx, y_input-dy), thickness = 2, color=(0,0,255))
    return angle/math.pi*180

num_list = [0, 0, 0, 0, 5]
p_list = [0,0,0,0,0]
angle_list = [0,0,0,0]
while(True):
    count_0 = 0
    count_4 = 0
    clock.tick()                    # Update the FPS clock.
    img = sensor.snapshot()         # Take a picture and return the image.
    #img.mean(1, threshold=True, offset=5, invert=True)
    #img.binary([(100,255)], invert = True)
    #img.erode(1)

    x00 = 91
    y00 = 4
    dx = 20
    dy = 20
    dis = 25
    p_thre = 0.95
    for i in range(0,5):
        class_num, pmax = mnist_run(img, dx, dy, dis,\
            x00 =x00, y00 = y00,\
            nnn=i)
        if pmax > p_thre:
            num_list[i] = class_num
            p_list[i] = pmax

    for i in range(0,5):
        if i == 4:
            x00 = x00
            dy = dy
        img.draw_rectangle((x00+dis*i,y00+i*0, dx, dy), color=255)
    R_list = []
    c_color = []
    x_list = [101+3, 175+2, 241, 263]
    y_list = [176-6, 193-6, 156-6, 84-6]

    angle_list[0] = search_col(x_list[0], y_list[0], img, width = 320, height = 240)
    angle_list[1] = search_col(x_list[1], y_list[1], img, width = 320, height = 240)
    angle_list[2] = search_col(x_list[2], y_list[2], img, width = 320, height = 240)
    angle_list[3] = search_col(x_list[3], y_list[3], img, width = 320, height = 240)
    print(num_list)
    print(p_list)
    #print(angle_list)
    R = 32
    img.draw_circle(x_list[0], y_list[0], R, color = (255, 0, 0), thickness = 2, fill = False)
    img.draw_circle(x_list[1], y_list[1], R, color = (255, 0, 0), thickness = 2, fill = False)
    img.draw_circle(x_list[2], y_list[2], R, color = (255, 0, 0), thickness = 2, fill = False)
    img.draw_circle(x_list[3], y_list[3], R, color = (255, 0, 0), thickness = 2, fill = False)

    # R-G-B 180-60-60
    r = 3
    img.draw_circle(x_list[0], y_list[0], r, color = (255, 255, 0), thickness = 1, fill = False)
    img.draw_circle(x_list[1], y_list[1], r, color = (255, 255, 0), thickness = 1, fill = False)
    img.draw_circle(x_list[2], y_list[2], r, color = (255, 255, 0), thickness = 1, fill = False)
    img.draw_circle(x_list[3], y_list[3], r, color = (255, 255, 0), thickness = 1, fill = False)
    utime.sleep_ms(250)
    #str(num_list[0])
    uart_B.write(str(0)+ str(0)+ str(0)+ str(0)+ str(6)+\
        '.' + str(int(angle_list[3]/36)) + str(int(angle_list[2]/36)) + str(int(angle_list[1]/36)) + str(int(angle_list[0]/36)))

    lcd.display(img)                # Display on LCD

    #lcd.draw_string(20,20,"%d: %.3f"%(max_index,pmax),lcd.WHITE,lcd.BLACK)
