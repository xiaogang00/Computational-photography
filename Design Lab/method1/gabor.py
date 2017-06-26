import math
import numpy as np
from matplotlib import pyplot as plt

def getFilter(theta,M,N):
    # filter configuration
    scale_bandwidth = np.abs(np.log2(0.55))
    angle_bandwidth = np.pi/8

    # x,y grid
    if N % 2 == 1:
        extentx = np.arange(-(N-1)/2, (N-1)/2 + 1)
    else:
        extentx = np.arange(-N/2, N/2)

    if M % 2 == 1:
        extenty = np.arange(-(M-1)/2, (M-1)/2 + 1)
    else:
        extenty = np.arange(-M/2, M/2)

    midx = int(N / 2)
    midy = int(M / 2)

    x, y = np.meshgrid(extentx, extenty)

    ## orientation component ##
    theta_0 = np.arctan2(y, x)
    number_orientations = 8
    center_angle = ((np.pi / number_orientations) * theta)

    costheta = np.cos(theta_0)
    sintheta = np.sin(theta_0)

    ds = sintheta * math.cos(center_angle) - costheta * math.sin(center_angle)
    dc = costheta * math.cos(center_angle) + sintheta * math.sin(center_angle)
    dtheta = np.arctan2(ds, dc)
    orientation_component = np.exp(-0.5 * (dtheta/angle_bandwidth)**2)

    ## frequency componenet ##
    # go to polar space
    raw = np.sqrt(x**2+y**2)
    # set origin to 1 as in the log space zero is not defined
    raw[midy, midx] = 1
    # go to log space
    raw = np.log2(raw)

    center_scale = np.log2(1.414)
    draw = raw-center_scale
    frequency_component = np.exp(-0.5 * (draw / scale_bandwidth)**2)

    # reset origin to zero (not needed as it is already 0?)
    frequency_component[midy, midx] = 0

    return frequency_component * orientation_component


if __name__ == "__main__":
    image = getFilter(np.pi* 1 /2, 128, 128)
    f1shift = np.fft.ifftshift(image)
    f1 = np.fft.ifft2(f1shift)
    f1 = np.fft.fftshift(f1)
    f1 = f1 * 10000
    print f1
    f1 = f1.real
    plt.imshow(f1), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()