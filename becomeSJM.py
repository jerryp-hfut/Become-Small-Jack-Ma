import cv2

# 加载人脸识别器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 加载需要贴在面部的图片
overlay_image = cv2.imread('overlay_image.jpg')

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头视频帧
    ret, frame = cap.read()

    # 将图片转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 标记人脸并贴图
    for (x, y, w, h) in faces:
        # 调整贴图尺寸与面部区域相同
        overlay_resized = cv2.resize(overlay_image, (w, h))

        # 在帧上进行图像融合
        for i in range(h):
            for j in range(w):
                frame[y + i, x + j] = overlay_resized[i, j]


        # 绘制人脸边框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

    # 显示结果
    cv2.imshow('Face Detection', frame)

    # 按下ESC键退出
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()