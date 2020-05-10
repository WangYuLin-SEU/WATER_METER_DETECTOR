import cv2
import random
import numpy as np 

def noise(img,snr):
    h=img.shape[0]
    w=img.shape[1]
    img1=img.copy()
    sp=h*w   # 计算图像像素点个数
    NP=int(sp*(1-snr))   # 计算图像椒盐噪声点个数
    for i in range (NP):
        randx=np.random.randint(1,h-1)   # 生成一个 1 至 h-1 之间的随机整数
        randy=np.random.randint(1,w-1)   # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random()<=0.5:   # np.random.random()生成一个 0 至 1 之间的浮点数
            img1[randx,randy]=0
        else:
            img1[randx,randy]=255
    return img1

def gen_img(num):
    img = np.zeros((28,28),dtype=np.uint8)
    x0 = random.randint(0,8)
    y0 = random.randint(24,28)
    think = random.randint(2,4)
    nn = random.randint(7,9)
    cv2.putText(img, str(num), (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=think)
    img = noise(img,0.1*nn)
    return img


k_list = []
for kk in range(300000):
    k_list.append(random.randint(0,9))
for kk in range(len(k_list)):
    img = gen_img(k_list[kk])
    cv2.imwrite('./dataset_mynum/'+str(k_list[kk])+'_'+str(kk)+'.bmp', img)