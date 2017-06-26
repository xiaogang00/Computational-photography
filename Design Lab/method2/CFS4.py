import cv2
from matplotlib import pyplot as plt
import numpy as np
from thin import ThinFilter



def CFSBackground(image):
    rows, cols = image.shape
    NewImage = np.zeros([rows, cols], np.uint8)
    NewImage[0: rows, 0: cols] = image.copy()
    label = 0
    for i in range(rows):
        for j in range(cols):
            if NewImage[i, j] == 255:
                stack = []
                stack.append([i, j])
                label += 10
                while len(stack) != 0:
                    curpixel = stack.pop()
                    x = curpixel[0]
                    y = curpixel[1]
                    if x + 1 < rows and x - 1 > 0 and y + 1 < cols and y - 1 > 0:
                        NewImage[x, y] = label
                        if NewImage[x, y - 1] == 255:
                            stack.append([x, y - 1])
                        if NewImage[x, y + 1] == 255:
                            stack.append([x, y + 1])
                        if NewImage[x - 1, y] == 255:
                            stack.append([x - 1, y])
                        if NewImage[x + 1, y] == 255:
                            stack.append([x + 1, y])
    value = []
    point = []
    for i in range(rows):
        for j in range(cols):
            if NewImage[i, j] in value:
                axis = value.index(NewImage[i, j])
                point.append([i, j, axis])
                continue
            else:
                value.append(NewImage[i, j])
                point.append([i, j, len(value) - 1])

    result = [[] for i in range(len(value))]
    for i in range(len(point)):
            result[point[i][2]].append([point[i][0], point[i][1]])
    print result[0]
    print len(result)
    plt.imshow(NewImage)
    plt.title('Thin Image')
    plt.show()
    return value, result


def Histogram(Oimage):
    rows, cols = Oimage.shape
    Histo = [0 for i in range(cols)]
    for i in range(cols):
        for j in range(rows):
            if Oimage[j, i] == 0:
                Histo[i] += 1

    Part = []
    temp = []
    begin = []
    flag = 1
    for i in range(cols):
        if Histo[i] != 0:
            if flag == 1:
                begin.append(i)
                flag = 0
            temp.append(Histo[i])
        else:
            flag = 1
            if temp != []:
                Part.append(temp)
            temp = []

    index = []
    for i in range(len(Part)):
        count = 0
        for j in range(len(Part[i])):
            if Part[i][j] >= 3:
                count += 1
            else:
                if count >= 25:
                    index.append([begin[i] + j - count, count])
                count = 0
                continue
    print index




def Remove(image):
    value, result = CFSBackground(image)
    print value
    print image
    Maxlength = len(result[0])
    Backindex = 0
    for i in range(len(result)):
        if len(result[i]) > Maxlength:
            Maxlength = len(result[i])
            Backindex = i
    print Backindex
    Foreindex = 0
    for i in range(len(value)):
        if value[i] == 0:
            Foreindex = i
            break
    rows, cols = image.shape
    print Foreindex
    for i in range(len(result)):
        if i != Backindex and i != Foreindex:
            for j in range(len(result[i])):
                image[result[i][j][0], result[i][j][1]]   = value[Backindex]
                if result[i][j][0] -1 > 0 and result[i][j][0]+1 <rows:
                    if  result[i][j][1] -1 > 0 and result[i][j][1]+1 < cols:
                        image[result[i][j][0]-1, result[i][j][1]] = 255
                        image[result[i][j][0]+1, result[i][j][1]] = 255
                        image[result[i][j][0], result[i][j][1]-1] = 255
                        image[result[i][j][0], result[i][j][1]+1] = 255

    return image



if __name__ == "__main__":
    img = cv2.imread('figure2.png')
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    SampleUpImage = cv2.pyrUp(GrayImage)
    ret, binimg = cv2.threshold(SampleUpImage, 165, 255, cv2.THRESH_BINARY)

    thinfilter = ThinFilter()
    thinimg = thinfilter.process(binimg)
    thinimg = Remove(thinimg)

    Result = Histogram(thinimg)

    plt.imshow(thinimg)
    plt.title('Thin Image')
    plt.show()