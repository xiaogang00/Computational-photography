import cv2
import math
import numpy as np
from matplotlib import pyplot as plt



# laplacian edge detection
#imgLap = cv2.Laplacian(imgGray, cv2.CV_8U)

#sobel edge detection
#sobelX = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0)
#sobelY = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1)

#sobelX = np.uint8(np.absolute(sobelX))
#sobelY = np.uint8(np.absolute(sobelY))

#sobelCombined = cv2.bitwise_or(sobelX, sobelY)

#canny = cv2.Laplacian(img_binary, cv2.CV_8U)
#imgAdapt = cv2.adaptiveThreshold(imgGray, 255,
#                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#coutours is the coordinate, and hierarchy is the relation for all the contours

#canny edge detection
def inflection(img_binary):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, img_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(img_binary, 50, 150)
    rows, cols = img_binary.shape
    contours, hierarchy = cv2.findContours(canny,   cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
    #cv2.imshow("img", img)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    code = []
    #print contours
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            if not(i == 0):
                if j == 0:
                    x = contours[i][j][0][0] - contours[i-1][len(contours[i-1]) - 1][0][0]
                    y = contours[i][j][0][1] - contours[i-1][len(contours[i-1]) - 1][0][1]
                else:
                    x = contours[i][j][0][0] - contours[i][j-1][0][0]
                    y = contours[i][j][0][1] - contours[i][j-1][0][1]
            else:
                if j == 0:
                    x = contours[i][j][0][0] - contours[len(contours) - 1][len(contours[i - 1]) - 1][0][0]
                    y = contours[i][j][0][1] - contours[len(contours) - 1][len(contours[i - 1]) - 1][0][1]

                else:
                    x = contours[i][j][0][0] - contours[i][j - 1][0][0]
                    y = contours[i][j][0][1] - contours[i][j - 1][0][1]

            if x == 1 and y == 0:
                code.append([i, j, 0])
            elif x == 1 and y == 1:
                code.append([i, j, 1])
            elif x == 0 and y == 1:
                code.append([i, j, 2])
            elif x == -1 and y == 1:
                code.append([i, j, 3])
            elif x == -1 and y == 0:
                code.append([i, j, 4])
            elif x == -1 and y == -1:
                code.append([i, j, 5])
            elif x == 0  and y == -1:
                code.append([i, j, 6])
            else:
                code.append([i, j, 7])

    cur = []
    for i in range(len(code)):
        if i == 0:
            cur.append(code[i][2] - code[len(code) - 1][2])
        else:
            cur.append(code[i][2] - code[i-1][2])
        if cur[i] > 8:
            cur[i] = 16 - cur[i]

    # plt.subplot(2, 2, 1), plt.imshow(img), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 2), plt.imshow(imgLap, cmap = 'gray'), plt.title('Laplacian Edge'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 3), plt.imshow(canny, cmap = 'gray'), plt.title('Canny Edge'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 4), plt.imshow(sobelCombined, cmap = 'gray'), plt.title('Sobel Edge'), plt.xticks([]), plt.yticks([])
    # plt.show()

    e = []
    for i in range(len(code)):
        if i == 0:
            e.append(cur[i] * (cur[len(code)-1] + cur[i] + cur[i+1]))
        elif i == len(code) - 1:
            e.append(cur[i] * (cur[i-1] + cur[i] + cur[0]))
        else:
            e.append(cur[i] * (cur[i-1]) + cur[i] + cur[i+1])

    t = 1
    num1 = 0
    num2 = 0
    for i in range(len(e)):
        e.append(e[i])

    e = map(abs, e)
    #print e
    e_start = [0 for i in range(500)]
    e_end   = [0 for i in range(500)]

    for i in range(len(e)):
        if i == 0:
            continue
        if e[i-1] < t and e[i] >= t:
            e_start[num1] = i
            num1 = num1 + 1
            i = i + 1
            if num1 > num2:
                for j in range(i, len(e), 1):
                    if e[j] >= t and e[j+1] <t:
                        e_end[num2] = j
                        i = j
                        num2 = num2 + 1
                        break
        if i > len(code):
            break

    e_max = []
    t = []

    for i in range(0, num1, 1):
        e_region = e[e_start[i]: e_end[i] + 1]
        e_max.append(max(e_region))
        k = e_region.index(max(e_region))
        t.append(k + e_start[i])
        if t[i] > len(code) - 1:
            t[i] = t[i] - len(code) + 1

    result = []
    for i in range(len(t)):
        x = contours[code[t[i]][0]][code[t[i]][1]][0][0]
        y = contours[code[t[i]][0]][code[t[i]][1]][0][1]
        result.append([x, y])

    #for i in range(rows):
    #    for j in range(cols):
    #        if [i, j] in result:
    #            cv2.circle(canny, (i, j), 1, 58, 1)

    #plt.imshow(canny, cmap = 'gray'), plt.title('Result'), plt.xticks([]), plt.yticks([])
    #plt.show()

    print len(t)
    return result

if __name__ == "__main__":
    img = cv2.imread('test3.png', cv2.IMREAD_COLOR)
    # cv2.imshow('image', img)
    cRange = 256
    rows, cols, channels = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #threshold, imgbinary = cv2.threshold(imgGray, 0, 255,
    #                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(img_binary, 50, 150)

    result1 = inflection(img_binary[:, 0:cols / 3])

    result2 = inflection(img_binary[:, cols / 3: cols * 2 / 3])
    for i in range(len(result2)):
        result2[i][0] = result2[i][0] + cols / 3

    result3 = inflection(img_binary[:, cols * 2 / 3: cols])
    for i in range(len(result3)):
        result3[i][0] = result3[i][0] + cols * 2 / 3

    img = cv2.imread('test3.png', cv2.IMREAD_COLOR)
    result1.extend(result2)
    result1.extend(result3)
    print len(result1)
    for i in range(cols):
        for j in range(rows):
            if [i, j] in result1:
                cv2.circle(img, (i, j), 1, (0, 255, 0), 1)


    #plt.imshow(canny, cmap = 'gray'), plt.title('Result'), plt.xticks([]), plt.yticks([])
    plt.imshow(img), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.waitKey(0)



