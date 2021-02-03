import cv2
import numpy as np
image = cv2.imread('source.jpg',cv2.IMREAD_COLOR)
YUV_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
Y, U, V = cv2.split(YUV_img)
#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('YUV_img_image.jpg',YUV_img)
Y = cv2.equalizeHist(Y)
cv2.imwrite('YUV_img_equalized.jpg',YUV_img)
edges = cv2.Canny(Y,100,200)
Y_np = np.float32(Y)
corners = cv2.goodFeaturesToTrack(Y_np,170,0.01,10)
corners = np.int0(corners)
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(edges, (x, y), 2, 255, -1)
cv2.imwrite('edges.jpg',edges)
dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
cv2.imwrite('dist.jpg',dist)
intergral_img = cv2.integral(Y)
intergral_img = intergral_img[1:,1:]
print(intergral_img.shape)
    # smoothed_filter
h, w = Y.shape
#for c in range(ch):
c = 0
for i in range(h):
    for j in range(w):
        kernel_size = int(round(dist[i][j]))
        if kernel_size % 2 == 0:
            kernel_size += 1
        if ((i <= kernel_size//2) or (j <= kernel_size//2) or (j > w - kernel_size//2 - 1)
                or (i > h - kernel_size//2 - 1) or (kernel_size <= 1)):
            continue
        _sum = (intergral_img[i-kernel_size//2-1][j-kernel_size//2-1]
                    - intergral_img[i-kernel_size//2-1][j+kernel_size//2]
                    + intergral_img[i+kernel_size//2][j+kernel_size//2]
                    - intergral_img[i+kernel_size//2][j-kernel_size//2-1])
                # whithout integral image
            # for l in range(i - kernel_size//2, i + kernel_size//2 + 1):
            #     for k in range(j - kernel_size//2, j + kernel_size//2 + 1):
            #         _sum += img[l][k]

        Y[i][j] = _sum//(kernel_size**2)
image = cv2.merge((Y, U, V))
image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
cv2.imwrite('res.jpg',image)
#cv2.namedWindow("res", cv2.WINDOW_NORMAL)
#cv2.imshow("res", image)
#cv2.waitKey(0)
cv2.destroyAllWindows()
