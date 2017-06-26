import cv2
from matplotlib import pyplot as plt
import numpy as np
from thin import ThinFilter

def CFS(image):
    rows, cols = image.shape
    NewImage = np.zeros([rows, cols], np.uint8)
    NewImage[0: rows, 0: cols] = image.copy()

    label = 0
    for i in range(rows - 2):
        for j in range(cols - 2):
            if NewImage[i, j] == 255:
                stack  = []
                stack.append([i, j])
                label += 10
                while len(stack) != 0:
                    curpixel = stack.pop()
                    x = curpixel[0]
                    y = curpixel[1]
                    if x+1 < rows and x-1 > 0 and y+1 < cols and y-1 > 0:
                        NewImage[x, y] = label
                        if NewImage[x, y-1] == 255:
                            stack.append([x, y-1])
                        if NewImage[x, y+1] == 255:
                            stack.append([x, y+1])
                        if NewImage[x-1, y] == 255:
                            stack.append([x-1, y])
                        if NewImage[x+1, y] == 255:
                            stack.append([x+1, y])

    value = []
    for i in range(rows):
        for j in range(cols):
            if NewImage[i, j] in value:
                #axis = value.index(image[i, j])
                #point.append([i, j, axis])
                continue
            else:
                value.append(NewImage[i, j])


    return value


def CFSBackground(image):
    rows, cols = image.shape
    NewImage = np.zeros([rows, cols], np.uint8)
    NewImage[0: rows, 0: cols] = image.copy()

    label = 0
    for i in range(rows - 2):
        for j in range(cols - 2):
            if NewImage[i, j] == 0:
                stack  = []
                stack.append([i, j])
                label += 10
                while len(stack) != 0:
                    curpixel = stack.pop()
                    x = curpixel[0]
                    y = curpixel[1]
                    if x+1 < rows and x-1 > 0 and y+1 < cols and y-1 > 0:
                        NewImage[x, y] = label
                        if NewImage[x, y-1] == 0:
                            stack.append([x, y-1])
                        if NewImage[x, y+1] == 0:
                            stack.append([x, y+1])
                        if NewImage[x-1, y] == 0:
                            stack.append([x-1, y])
                        if NewImage[x+1, y] == 0:
                            stack.append([x+1, y])
                        if NewImage[x + 1, y + 1] == 0:
                            stack.append([x + 1, y + 1])
                        if NewImage[x + 1, y - 1] == 0:
                            stack.append([x + 1, y - 1])
                        if NewImage[x - 1, y + 1] == 0:
                            stack.append([x - 1, y + 1])
                        if NewImage[x - 1, y - 1] == 0:
                            stack.append([x - 1, y - 1])

    for i in range(rows):
        for j in range(cols):
            if NewImage[i, j] == 255:
                NewImage[i, j] = 0

    value = []
    for i in range(rows):
        for j in range(cols):
            if NewImage[i, j] in value:
                continue
            else:
                value.append(NewImage[i, j])

    return value

def CrossDetection(image):
    rows, cols = image.shape
    print rows, cols
    NewImage = np.zeros([rows, cols], np.uint8)
    NewImage[0: rows, 0: cols] = image.copy()
    count = 0
    Width = 10
    Height = 16
    Result = []


    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            if j+Width < cols and i+Height < rows:
                flag = 0
                length1 = 1
                length2 = 12
                value1 = CFSBackground(NewImage[i:i + Height, j:j + Width])
                if sum(sum(NewImage[i:i+length1, j:j+length2])) >= 255 * length1 * length2:
                    if sum(sum(NewImage[i:i+length1, j+Width-length2:j+Width])) >= 255 * length1 * length2:
                        if sum(sum(NewImage[i+Height-length1:i+Height, j:j+length2])) >= 255 * length1 * length2:
                            if sum(sum(NewImage[i+Height-length1:i+Height, j+Width-length2:j+Width])) \
                                    >= 255 * length1 * length2:
                                flag = 1

                NewImage[i, j:j+Width] = 0
                NewImage[i:i+Height, j] = 0
                NewImage[i+Height-1, j:j+Width] = 0
                NewImage[i:i+Height, j+Width-1] = 0
                value = CFS(NewImage[i:i+Height, j:j+Width])
                if len(value) == 5 and i < rows / 3 and len(value1) == 2 and flag == 1:
                    label = 1
                    print [i, j]
                    for k in range(len(Result)):
                        if np.abs(i - Result[k][0]) < 6 and np.abs(j - Result[k][1]) < 4:
                            label = 1
                            break
                    if label == 1:
                        Result.append([i, j])
                        count += 1

                NewImage[i, j:j+Width] = image[i, j:j+Width]
                NewImage[i:i+Height, j] = image[i:i+Height, j]
                NewImage[i+Height-1, j:j+Width] = image[i+Height-1, j:j+Width]
                NewImage[i:i+Height, j+Width-1] = image[i:i+Height, j+Width-1]
    print count

    return image, Result


if __name__ == "__main__":
    img = cv2.imread('figure4.png')
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    SampleUpImage = cv2.pyrUp(GrayImage)
    ret, binimg = cv2.threshold(SampleUpImage, 165, 255, cv2.THRESH_BINARY)

    thinfilter = ThinFilter()
    thinimg = thinfilter.process(binimg)
    image, Result = CrossDetection(thinimg)
    rows, cols = image.shape

    Width = 10
    Height = 16
    for k in range(len(Result)):
        i = Result[k][0]
        j = Result[k][1]
        image[i, j:j + Width] = 0
        image[i:i + Height, j] = 0
        image[i + Height - 1, j:j + Width] = 0
        image[i:i + Height, j + Width - 1] = 0

    plt.imshow(image)
    plt.title('Thin Image')
    plt.show()