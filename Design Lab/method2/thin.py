#encoding=utf-8
import numpy as np
import cv2 
from matplotlib import pyplot as plt
from numpy import *
from matplotlib.axis import GRIDLINE_INTERPOLATION_STEPS

class ThinFilter:

    def _private_step1Scan(self,image,flagmap):
        stop = True;
        height = image.shape[0];
        width = image.shape[1];
        for row in range(1,height -1):
            for col in range(1,width -1):
                con2 = 0
                p1 = image[row][col];
                if p1 == 255:
                    continue
                p9 = 1 if image[row-1][col-1] == 0 else 0
                p2 = 1 if image[row-1][col] == 0 else 0
                p3 = 1 if image[row-1][col+1] == 0 else 0
                p8 = 1 if image[row][col-1] == 0 else 0
                p4 = 1 if image[row][col+1] == 0 else 0
                p7 = 1 if image[row+1][col-1] == 0 else 0
                p6 = 1 if image[row+1][col] == 0 else 0
                p5 = 1 if image[row+1][col+1] == 0 else 0
                if p2==0 and p3==1:
                    con2+=1
                if p3==0 and p4==1:
                    con2+=1
                if p4==0 and p5==1:
                    con2+=1
                if p5==0 and p6==1:
                    con2+=1
                if p6==0 and p7==1:
                    con2+=1
                if p7==0 and p8==1:
                    con2+=1
                if p8==0 and p9==1:
                    con2+=1
                if p9==0 and p2==1:
                    con2+=1
                con1 = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                con3 = p2 * p4 * p6
                con4 = p4 * p6 * p8
                
                if con2==1 and con1>=2 and con1<=6 and con3 ==0 and con4 ==0 :
                    flagmap[row][col] = 1
                    stop = False 
        
        return stop
    
    def _private_step2Scan(self,image,flagmap):
        stop = True;
        height = image.shape[0];
        width = image.shape[1];
        for row in range(1,height -1):
            for col in range(1,width -1):
                con2 = 0
                p1 = image[row][col]
                if p1 == 255:
                    continue
                p9 = 1 if image[row-1][col-1] == 0 else 0
                p2 = 1 if image[row-1][col] == 0 else 0;
                p3 = 1 if image[row-1][col+1] == 0 else 0;
                p8 = 1 if image[row][col-1] == 0 else 0;
                p4 = 1 if image[row][col+1] == 0 else 0;
                p7 = 1 if image[row+1][col-1] == 0 else 0;
                p6 = 1 if image[row+1][col] == 0 else 0;
                p5 = 1 if image[row+1][col+1] == 0 else 0;
                if p2==0 and p3==1:
                    con2+=1
                if p3==0 and p4==1:
                    con2+=1
                if p4==0 and p5==1:
                    con2+=1
                if p5==0 and p6==1:
                    con2+=1
                if p6==0 and p7==1:
                    con2+=1
                if p7==0 and p8==1:
                    con2+=1
                if p8==0 and p9==1:
                    con2+=1
                if p9==0 and p2==1:
                    con2+=1
                con1 = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                con3 = p2 * p4 * p8
                con4 = p2 * p6 * p8
                
                if con2==1 and con1>=2 and con1<=6 and con3 ==0 and con4 ==0 :
                    flagmap[row][col] = 1
                    stop = False 
        
        return stop
    
    def _private_deletewithFlag(self,image,flagmap):
        height = image.shape[0];
        width = image.shape[1];
        for row in range(0,height):
            for col in range(0,width):
                if flagmap[row][col] == 1:
                    image[row][col] = 255
                    flagmap[row][col] = 0 
                    
    def process(self,image):
        height = image.shape[0];
        width = image.shape[1];
        flagmap = zeros((height,width), dtype=int32)
        stop = False
        while stop == False:
            s1 = self._private_step1Scan(image, flagmap)
            self._private_deletewithFlag(image, flagmap)
            s2 = self._private_step2Scan(image, flagmap)
            self._private_deletewithFlag(image, flagmap)
            if s1 and s2:
                stop =True
        return image
                
