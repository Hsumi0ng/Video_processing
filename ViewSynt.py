import imageio
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def show(img):
    plt.imshow(img,'gray')
    plt.show()

def show_double(img1,img2,title1,title2):
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1,'gray')
    plt.title(title1)
    plt.subplot(122)
    plt.imshow(img2,'gray')
    plt.title(title2)
    plt.show()

def filterLayer(img):
    img=cv.medianBlur(img,3)
    img=img/16.0
    return img
########################initialization#############################
imgC = cv.imread('RV_L.bmp')
imgL = cv.imread('RV_L.bmp',0)
imgR = cv.imread('RV_R.bmp',0)
imgL1=cv.imread('RV_L.bmp',0)
imgR1=cv.imread('RV_R.bmp',0)
H,W=imgL1.shape
show(imgL1)
########################disparity computation########################
windowsize=15
channel=1
numdisparities=16*2
stereoL = cv.StereoSGBM_create(0,numdisparities,windowsize,8*channel*windowsize**2,32*channel*windowsize**2,-1,15,0,10,1,1)
disparity_L = stereoL.compute(imgL1,imgR1)
stereoR = cv.ximgproc.createRightMatcher(stereoL)
disparity_R = abs(stereoR.compute(imgR1,imgL1))
for i in range(H):
    for j in range(numdisparities):
        disparity_L[i][0:numdisparities]=disparity_L[i][numdisparities+1]
        disparity_R[i][W-numdisparities:W+1]=disparity_R[i][W-numdisparities-1]
#######################filter########################################
disparity_L=filterLayer(disparity_L)
disparity_R=filterLayer(disparity_R)
title1='L2R Disparity map'
title2='R2L Disparity map'
show_double(disparity_L,disparity_R,title1,title2)
#######################LRC-checking##################################
Occlusion=np.zeros_like(disparity_L)
for i in range(H):
    for j in range(W):
        dispL=disparity_L[i][j]
        if j-dispL>=W:
            Occlusion[i][j]=0
            continue
        dispR=disparity_R[i][j-round(dispL)]
        if dispL==0 and dispR==0:
            Occlusion[i][j]=10
            continue
        if dispL==-1:
            Occlusion[i][j]=-1
            continue
        if abs(dispL-dispR)<1:Occlusion[i][j]=disparity_L[i][j]
        else:Occlusion[i][j]=0
#show(Occlusion)
DisparityL_LRC=np.zeros_like(disparity_L)
for i in range(H):
    for j in range(W):
        if Occlusion[i][j]>0:
            DisparityL_LRC[i][j]=disparity_L[i][j]
            continue
        else:
            pl=0
            pr=0
            left=1
            right=1
            flagL=1
            flagR=1
            while(flagL):
                if j-left<=0:
                    flagL=0
                    pl=1000
                    break
                pl=disparity_L[i][j-left]
                if pl>0:
                    flagL=0
                else:left=left+1
            while(flagR):
                if j+right==W:
                    flagR=0
                    pr=1000
                    break
                pr=disparity_L[i][j+right]
                if pr>0:
                    flagR=0
                else:right=right+1
        if Occlusion[i][j]==0:
            DisparityL_LRC[i][j]=min(pl,pr)
        elif Occlusion[i][j]==-1:
            DisparityL_LRC[i][j]=max(pl,pr)
##############################hollow-refinement#############################\
DisparityL_LRC=cv.medianBlur(DisparityL_LRC.astype(np.uint8),5)
for i in range(H):
    for j in range(1,W):
        if DisparityL_LRC[i][j]>200:
            DisparityL_LRC[i][j]=DisparityL_LRC[i][j-1]
    for k in range(1,W):
        if DisparityL_LRC[i][W-k-1]>200:
            DisparityL_LRC[i][W-k-1]=DisparityL_LRC[i][W-k]
#show(DisparityL_LRC)
Mask=np.zeros_like(disparity_L)
Mask=0.7*DisparityL_LRC+0.3*imgL
#show(Mask)
DMask=DisparityL_LRC.copy()
DMask[DMask<=20]=0
DMask[DMask>20]=255
DMask=255-DMask
show_double(imgL,DMask,'1','2')
DMasked=cv.bitwise_and(DMask,imgL)
show(DMask)
show(DMasked)
#######################edge-pixel-refinement#####################
sobel_x = cv.Sobel(DMasked, -1, 1, 0, ksize=3)
sobel_y = cv.Sobel(DMasked, -1, 0, 1, ksize=3)
edge=cv.addWeighted(sobel_x,0.5,sobel_y,0.5,0)
Disp_B=DisparityL_LRC.copy()
edge[edge<70]=0
edge[edge>=70]=255
for i in range(1,H-1):
    for j in range(1,W-1):
        if edge[i][j]==0:continue
        else:
            if abs(sum(DMask[i][j-5:j-1])-sum(DMask[i][j+1:j+5]))>100:edge[i][j]=0
show_double(DisparityL_LRC,edge,"disp","edge")
ws=10
for i in range(0,H):
    for j in range (ws,W-2*ws):
        if edge[i][j]==0:continue
        else:DisparityL_LRC[i][j:j+ws]=np.median(DisparityL_LRC[i][j+ws:j+2*ws])
DisparityL_LRC=max(DisparityL_LRC.flatten())-DisparityL_LRC
Disp_B=max(Disp_B.flatten())-Disp_B
show_double(DisparityL_LRC,Disp_B,"1","2")
DMasked=cv.subtract(Disp_B,DisparityL_LRC)
d=cv.subtract(DisparityL_LRC,Disp_B)
show(DMasked)
DMask=DisparityL_LRC.copy()
DMask[DMask<10]=0
DMask[DMask>10]=255
show_double(imgL,DMask,'1','2')
DMasked=cv.bitwise_and(DMask,imgL)
show(DMask)
show(DMasked)
###########################DOF######################################
'''
Focus_len=12
Baseline=30
depth_L=np.zeros_like(disparity_L)
for i in range(H):
    for j in range(W):
        if disparity_L[i][j]==0:continue
        else:depth_L[i][j] = Focus_len*Baseline/disparity_L[i][j]
show(255-depth_L)
'''
############################3D-construction#########################
number=50
for i in range(number):
    visual_Synthesis=np.zeros_like(imgC)
    print(i)
    for j in range(H):
        for k in range(W):
            if DMask[j][k]==0:
                disp_VS=DisparityL_LRC[j][k]
                if k+round(i*1/number*disp_VS)>=W:visual_Synthesis[j][k]=imgC[j][k]
                else:visual_Synthesis[j][k]=imgC[j][k-round(i*1/number*disp_VS)]
            else:continue
    for j in range(H):
        for k in range(W):
            if DMask[j][k]==255:
                disp_VS=DisparityL_LRC[j][k]
                if k+round(i*1/number*disp_VS)>=W:visual_Synthesis[j][k]=imgC[j][k]
                else:visual_Synthesis[j][k]=imgC[j][k-round(i*1/number*disp_VS)]
    cv.cvtColor(visual_Synthesis,cv.COLOR_BGR2RGB)
    cv.imwrite("C:/Users/PC/Desktop/assignment4/Rhine_valley/output/"+str(i)+".bmp",visual_Synthesis)
################################gif########################################################

with imageio.get_writer(uri='test.gif',mode='I',fps=24) as writter:
    for i in range(number):
        writter.append_data(imageio.imread(f'C:/Users/PC/Desktop/assignment4/Rhine_valley/output/{i}.bmp'))
    for i in range(number):
        writter.append_data(imageio.imread(f'C:/Users/PC/Desktop/assignment4/Rhine_valley/output/{50-1-i}.bmp'))
with imageio.get_writer(uri='test1.gif',mode='I',fps=24) as writter:
    for i in range(25):
        writter.append_data(imageio.imread(f'C:/Users/PC/Desktop/assignment4/Rhine_valley/output/{i}.bmp'))
    for i in range(25):
        writter.append_data(imageio.imread(f'C:/Users/PC/Desktop/assignment4/Rhine_valley/output/{25-1-i}.bmp'))
