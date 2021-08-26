from ex2_utils import *

"""
All the picture scale between 0-1
"""


def test_conv1D():
    signal = np.array([1, 2, 3, 4, 5])
    kernel = np.array([1, 2, 3])
    print("numpy: ", np.convolve(signal, kernel, "full"))
    print("mine: ", conv1D(signal, kernel))


def test_conv2D():
    signal2D = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)/255
    kernel = np.ones(shape=(3, 3))  # mean filter
    their = cv2.filter2D(signal2D, -1, kernel/np.sum(kernel), borderType=cv2.BORDER_REPLICATE)
    mine = conv2D(signal2D, kernel)
    plt.gray()
    plt.imshow(their)
    plt.show()
    plt.imshow(mine)
    plt.show()
    # check the mean distance between openCV and my implementation
    # i got very low distance
    # imcol = signal2D.shape[1]
    # imrow = signal2D.shape[0]
    # res = (np.abs(np.subtract(mine, their)).sum())/(imrow*imcol)
    # print(res)


def test_convDerivative():
    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    directions, magnitude, im_derive_x, im_derive_y = convDerivative(img)
    plt.gray()
    plt.subplot(2, 2, 1), plt.imshow(img)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(magnitude)
    plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(im_derive_x)
    plt.title('Derivative X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(im_derive_y)
    plt.title('Derivative Y'), plt.xticks([]), plt.yticks([])
    plt.show()

def test_blurImage():
    img = cv2.imread("beach.jpg", cv2.IMREAD_GRAYSCALE)/255
    kernel_size = np.array([10, 10])
    blur1 = blurImage1(img, kernel_size)*255
    blur2 = blurImage2(img, kernel_size)
    plt.gray()
    plt.subplot(1, 2, 1), plt.imshow(blur1)
    plt.title('MyBlur'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(blur2)
    plt.title('CV_Blur'), plt.xticks([]), plt.yticks([])
    plt.show()

def test_edgeDetectionSobel():
    img = cv2.imread("coins.jpg", cv2.IMREAD_GRAYSCALE)/255
    their, mine = edgeDetectionSobel(img, 0.5)
    plt.gray()
    plt.subplot(2, 2, 1), plt.imshow(img)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(their)
    plt.title('their'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(mine)
    plt.title('mine'), plt.xticks([]), plt.yticks([])
    plt.show()

def test_edgeDetectionZeroCrossingSimple():
    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    ans = edgeDetectionZeroCrossingSimple(img)
    plt.gray()
    plt.imshow(ans)
    plt.show()

def test_edgeDetectionZeroCrossingLOG():
    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    plt.gray()
    # plt.imshow(img)
    # plt.show()
    ans = edgeDetectionZeroCrossingLOG(img)
    plt.imshow(ans)
    plt.show()


def test_edgeDetectionCanny():
    img = cv2.imread("Hands.jpg", cv2.IMREAD_GRAYSCALE)
    img = img/255
    canny, mine = edgeDetectionCanny(img, 0.65, 0.8)
    plt.gray()
    plt.subplot(2, 2, 1), plt.imshow(img)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(canny)
    plt.title('Open CV Canny'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(mine)
    plt.title('Mine canny'), plt.xticks([]), plt.yticks([])
    plt.show()

def test_hough_circles():
    img = cv2.imread("Circles.jpg", cv2.IMREAD_GRAYSCALE)/255
    min_radius = 30
    max_radius = 60
    circle_list = houghCircle(img, min_radius, max_radius)
    # draw the circle in the original image
    # Draw shortlisted circles on the output image
    for vertex in circle_list:
        cv2.circle(img, (vertex[1], vertex[0]), vertex[2], (0, 0, 255), 3)
    cv2.imshow('Circle Detected Image', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # test_conv1D()
    # test_conv2D()
    test_convDerivative()
    # test_blurImage()
    # test_edgeDetectionSobel()
    # test_edgeDetectionZeroCrossingSimple()
    # test_edgeDetectionZeroCrossingLOG()
    # test_edgeDetectionCanny()
    # test_hough_circles()
