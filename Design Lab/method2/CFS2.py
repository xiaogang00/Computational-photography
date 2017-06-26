import cv2
from matplotlib import pyplot as plt
from numpy import *
from thin import ThinFilter


def CFSBackground(image):
    rows, cols = image.shape
    label = 0
    for i in range(rows - 2):
        for j in range(cols - 2):
            if image[i, j] == 255:
                stack = []
                stack.append([i, j])
                label += 10
                while len(stack) != 0:
                    curpixel = stack.pop()
                    x = curpixel[0]
                    y = curpixel[1]
                    if x + 1 < rows and x - 1 > 0 and y + 1 < cols and y - 1 > 0:
                        image[x, y] = label
                        if image[x, y - 1] == 255:
                            stack.append([x, y - 1])
                        if image[x, y + 1] == 255:
                            stack.append([x, y + 1])
                        if image[x - 1, y] == 255:
                            stack.append([x - 1, y])
                        if image[x + 1, y] == 255:
                            stack.append([x + 1, y])
    value = []
    point = []
    for i in range(rows):
        for j in range(cols):
            if image[i, j] in value:
                axis = value.index(image[i, j])
                point.append([i, j, axis])
                continue
            else:
                value.append(image[i, j])
                point.append([i, j, len(value) - 1])

    result = [[] for i in range(len(value))]
    for i in range(len(point)):
            result[point[i][2]].append([point[i][0], point[i][1]])

    return image, value, result


def RemoveLarge(image, value, result):
    Maxindex = 0
    MaxLen = len(result[0])
    for i in range(len(result)):
        if len(result[i]) > MaxLen:
            Maxindex = i

    index = []
    for i in range(len(result)):
        print len(result[i])
        if len(result[i]) > 1500:
            index.append(i)
            for j in range(len(result[i])):
                image[result[i][j][0], result[i][j][1]] = value[Maxindex]
    return image, index


if __name__ == "__main__":
    img = cv2.imread('test.png')
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    SampleUpImage = cv2.pyrUp(GrayImage)
    ret, binimg = cv2.threshold(SampleUpImage, 165, 255, cv2.THRESH_BINARY)

    thinfilter = ThinFilter()
    thinimg = thinfilter.process(binimg)

    image, value, result = CFSBackground(thinimg)
    image, index =RemoveLarge(image, value, result)
    #index返回的是其中可能是两个数字相接的情况的在result中的坐标，不属于index中下标的result中的坐标值是有效的字母
    plt.imshow(image)
    plt.title('Thin Image')
    plt.show()