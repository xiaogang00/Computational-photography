import cv2
from matplotlib import pyplot as plt
from numpy import *
from thin import ThinFilter


def CFS_modified(image, result ,value):
    rows, cols = image.shape
    label = 100
    for i in range(len(result)):
        for j in range(len(result[i])):
            x = result[i][j][0]
            y = result[i][j][1]
            if (image[x, y] == value[i]):
                stack = []
                stack.append([x, y])
                label += 10
                count = 1
                while len(stack) != 0:
                    curpixel = stack.pop()
                    x = curpixel[0]
                    y = curpixel[1]
                    if count != 1:
                        image[x, y] = label
                    if x - 1 < rows and x - 1 >= 0:
                        if image[x - 1, y] == value[i]:
                            stack.append([x - 1, y])
                            count += 1
                    if x + 1 < rows and x + 1 >= 0:
                        if image[x + 1, y] == value[i]:
                            stack.append([x + 1, y])
                            count += 1

    for i in range(rows):
        for j in range(cols):
            if not(image[i, j] in value):
                image[i, j] = 0
            else:
                image[i, j] = 255

    return image


def CFS(image):
    rows, cols = image.shape
    label = 0
    for i in range(rows - 2):
        for j in range(cols - 2):
            if image[i, j] == 0:
                stack  = []
                stack.append([i, j])
                label += 10
                while len(stack) != 0:
                    curpixel = stack.pop()
                    x = curpixel[0]
                    y = curpixel[1]
                    if x+1 < rows and x-1 >0 and y+1 < cols and y-1>0:
                        image[x, y] = label
                        if image[x, y-1] == 0:
                            stack.append([x, y-1])
                        if image[x, y+1] == 0:
                            stack.append([x, y+1])
                        if image[x-1, y] == 0:
                            stack.append([x-1, y])
                        if image[x+1, y] == 0:
                            stack.append([x+1, y])
                        if image[x+1, y+1] == 0:
                            stack.append([x+1, y+1])
                        if image[x+1, y-1] == 0:
                            stack.append([x+1, y-1])
                        if image[x-1, y+1] == 0:
                            stack.append([x-1, y+1])
                        if image[x-1, y-1] == 0:
                            stack.append([x-1, y-1])

    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 255:
                image[i, j] = 0

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
                point.append([i, j, len(value)-1])

    result = [[] for i in range(len(value))]
    for i in range(len(point)):
        if point[i][2] != 0:
            #print point[i][2], point[i][1], point[i][0]
            result[point[i][2]].append([point[i][0], point[i][1]])

    return image, value, result


def LineRemove(image):
    im, value, result = CFS(image)
    for i in range(len(result)):
        length = len(result[i])
        if length < 20:
            for j in range(len(result[i])):
                image[result[i][j][0]][result[i][j][1]] = 0

    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            if image[i, j] != 0:
                image[i, j] = 0
            else:
                image[i, j] = 255

    return image


def DotDetection(image, result1):

    minlen = len(result1[1])
    flag = 1
    for i in range(len(result1)):
        if i != 0:
            length = len(result1[i])
            if length < minlen:
                minlen = length
                flag = i

    x = 0
    y = 0
    for i in range(len(result1[flag])):
        x += result1[flag][i][0]
        y += result1[flag][i][1]

    meanX = x // len(result1[flag])
    meanY = y // len(result1[flag])

    return meanX, meanY, flag


def FindLine(image, result, meanX, meanY):
    distance = [0 for i in range(len(result))]
    for i in range(len(result)):
        if i != 0:
            meanx = 0
            meany = 0
            for j in range(len(result[i])):
                meanx += result[i][j][0]
                meany += result[i][j][1]
            meanx = meanx // len(result[i])
            meany = meany // len(result[i])
            temp = math.sqrt((meanx - meanX)*(meanx - meanX) + (meany - meanY)*(meany - meanY))
            distance[i] = temp

    flag = 1
    temp_distance = distance[1]
    for i in range(len(distance)):
        if i != 0:
            if distance[i] <temp_distance:
                temp_distance = distance[i]
                flag = i

    for i in range(len(result)):
        if i == flag:
            for j in range(len(result[i])):
                image[result[i][j][0], result[i][j][1]] = 100
        else:
            for j in range(len(result[i])):
                image[result[i][j][0], result[i][j][1]] = 10

    return result[flag], image


def lineDetection(image):
    rows, cols = image.shape
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    image = cv2.line(image, (cols - 1, righty), (0, lefty), 255, 2)
    return image

if __name__ == "__main__":
    img = cv2.imread('figure2.png')
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    SampleUpImage = cv2.pyrUp(GrayImage)
    ret, binimg = cv2.threshold(SampleUpImage, 165, 255, cv2.THRESH_BINARY)


    thinfilter = ThinFilter()
    thinimg = thinfilter.process(binimg)


    image, value, result = CFS(thinimg)

    meanX, meanY, flag = DotDetection(image, result)
    print meanX, meanY


    image2 = CFS_modified(image, result, value)

    image3 = LineRemove(image2)


    image4, value, result = CFS(image3)
    point, FinalImage = FindLine(image4, result, meanX, meanY)

    plt.imshow(FinalImage)
    plt.title('Thin Image')
    plt.show()