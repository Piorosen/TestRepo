import threading
from tkinter import *
import numpy as np
from PIL import Image
import detection as p
import cv2

# 블러 디텍션
# https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
def BlurDetect(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# 히스토그램 평활화
def EqualizeImage(image):
    return cv2.equalizeHist(image)

# 이미지 유사도 측정
# 장경식 교수님 코드 참조.
def SimilityImage(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

#
def ImageRotate(img):
    imgOrigin = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html
    thresh = cv2.adaptiveThreshold(
        img,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )

    # ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # https://www.wake-up-neo.com/ko/python/opencv%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EC%97%AC-%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%97%90%EC%84%9C-%EC%82%AC%EA%B0%81%ED%98%95%EC%9D%98-%EC%A4%91%EC%8B%AC%EA%B3%BC-%EA%B0%81%EB%8F%84-%EA%B0%90%EC%A7%80/823074665/
    contours, e = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("Not Found")
    len(contours[0])
    cntsFirst = -1

    for item in range(len(contours)):
        rect = cv2.minAreaRect(contours[item])
        if abs(rect[2]) > 0.5 and rect[1][0] > 50 and rect[1][1] > 20:
            cntsFirst = item
            break

    print(cntsFirst)
    cnts = contours[cntsFirst]

    plate_rect = cv2.minAreaRect(cnts)
    print(plate_rect)


    # 45, -45 데이터는 잘못 각도가 계산이 된것이므로, 수정해주어야 합니다.
    # 경험 상
    image_rect = plate_rect
    if plate_rect[2] > 45 and plate_rect[2] < 135:
        image_rect = (plate_rect[0], plate_rect[1], plate_rect[2] - 90)
    elif plate_rect[2] < -45 and plate_rect[2] > -135:
        image_rect = (plate_rect[0], plate_rect[1], plate_rect[2] - 90)

    plate_box = np.int0(cv2.boxPoints(plate_rect))
    image_box = np.int0(cv2.boxPoints(image_rect))

    # https://pybasall.tistory.com/135
    avg = plate_box.mean(axis=0)
    image_center = (avg[0], avg[1])
    avg
    # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point

    plate_rot_mat = cv2.getRotationMatrix2D(image_center, plate_rect[2], 1.0)
    image_rot_mat = cv2.getRotationMatrix2D(image_center, image_rect[2], 1.0)

    pts = np.int0(cv2.transform(np.array([plate_box]), plate_rot_mat))[0]
    rot_image = cv2.warpAffine(imgOrigin, image_rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    pts = np.int0(cv2.transform(np.array([image_box]), plate_rot_mat))[0]
    rot_image.shape

    pts.sort(axis=0)
    pts[3][1]
    # crop

    img_crop = rot_image[pts[0][1]:pts[3][1], pts[0][0]:pts[3][0]]
    img_crop.shape
    return img_crop

def ImageRecognize(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_equal = EqualizeImage(img_gray)

    img_thresh = cv2.adaptiveThreshold(
        img_equal,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=9,
        C=0
    )
    img_blurred = cv2.GaussianBlur(img_thresh, ksize=(5, 5), sigmaX=0)
    _, img_the = cv2.threshold(img_blurred, 80, 255, cv2.THRESH_BINARY)
    return img_the

# 카메라 1번을 사용합니다.
cap = cv2.VideoCapture(0)

suspend = True
previousImage = None

def WebCamOff():
    global suspend
    suspend = True


def webCamThread(uiRoot):
    global previousImage
    global isFirst
    global suspend
    global cap
    while not suspend:
        ret, frame = cap.read()
        score = BlurDetect(frame)
        if score > 100:
            uiRoot.UpdateImage(frame, None, None)
            print("Blur Score : " + str(score))
            uiRoot.StatusLabel("블러 점수 100점 이하")
            continue
        elif not (previousImage is None):
            score = SimilityImage(previousImage, frame)
            if score > 0.95:
                uiRoot.UpdateImage(frame, None, None)
                print("Similar Score : " + str(score))
                uiRoot.StatusLabel("유사도 95%이상")
                continue
        previousImage = frame

        uiRoot.StatusLabel("YOLO 테스트")
        result = p.predict(frame)
        for item in range(len(result)):
            if result[item].get_score() > 0.45:
                (xmin, ymin, xmax, ymax) = result[item].get_rect()
                print((xmin, ymin, xmax, ymax))
                img = frame[ymin:ymax, xmin:xmax].copy()
                img_crop = ImageRotate(img)
                if img_crop.size != 0:
                    img_adaptive = ImageRecognize(img_crop)
                    uiRoot.UpdateImage(frame, img, img_adaptive)
                    cv2.waitKey(27)


def WebCamOn(uiRoot):
    global suspend
    if not suspend:
        return
    suspend = False
    my = threading.Thread(target=webCamThread, args=(uiRoot,))
    my.start()


