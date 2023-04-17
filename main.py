import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui
import math
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import time
import heatmap
import Sticker
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QMovie

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def findDistance(p1, p2):
    # 눈과 눈 사이
    x1, y1 = p1
    # (x1, y1)
    x2, y2 = p2
    # 거리 찾기임
    length = math.hypot(x2 - x1, y2 - y1)

    return length


app = QtWidgets.QApplication(sys.argv)

# 미디어 파이프를 mp로 가져오기
mp_face_mesh = mp.solutions.face_mesh

pupil_model = load_model('model/pupil.h5')

Init = True
# ------------------------
# 비디오 키기(웹캠)
cap = cv.VideoCapture(0)
ad = cv.VideoCapture('ad_6.mp4')
fps = round(cap.get(cv.CAP_PROP_FPS))
# 얼굴 점찍기
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,  # 얼굴 감지모델에 대한 최소 감지 신뢰도
        min_tracking_confidence=0.5  # 랜드마크 추적기 모델의 경우 랜드마크 추격을 위한 최소 신뢰도
) as face_mesh:
    while True:
        if Init:
            Center = False
            RightUp = False
            RightDown = False
            LeftUp = False
            LeftDown = False
            Start = False
            CenterUp = False
            CenterDown = False
            Left = False
            Right = False
            Eye_Tracking = False
            MoveCenter = False

            Show = True
            Video = False
            pyautogui.FAILSAFE = False
            Heatmap = False
            Cap = True

            Center_nose = False
            RightUp_nose = False
            RightDown_nose = False
            LeftUp_nose = False
            LeftDown_nose = False
            Start_nose = False
            CenterUp_nose = False
            CenterDown_nose = False
            Left_nose = False
            Right_nose = False

            Dot = False
            Dot2 = False
            Circle = False
            Walk = False
            Image = False
            TwoImage = False
            Clock = False
            Clock_HeatMap = False

            monitorWidth, monitorHeight = pyautogui.size()

            center_location = []
            rightup_location = []
            rightdown_location = []
            leftup_location = []
            leftdown_location = []
            centerup_location = []
            centerdown_location = []
            left_location = []
            right_location = []

            center_location_nose = []
            rightup_location_nose = []
            rightdown_location_nose = []
            leftup_location_nose = []
            leftdown_location_nose = []
            centerup_location_nose = []
            centerdown_location_nose = []
            left_location_nose = []
            right_location_nose = []

            # 좌표값
            location = []
            Video_location = []
            Clock_location = []

            pt = np.array([0, 0], dtype=np.int32)

            dot_x = int(monitorWidth / 2)
            dot_y = int(monitorHeight / 2)
            dot_nose_x = int(monitorWidth / 2)
            dot_nose_y = int(monitorHeight / 2)

            count = 0
            prev_time = 0
            left_top_clock = 0
            left_down_clock = 0
            right_top_clock = 0
            right_down_clock = 0

        Init = False

        ret, frame = cap.read()
        if Video:
            suc, video_frame = ad.read()
            if not suc:
                cv.destroyAllWindows()
                for i in range(count):
                    Video_location.append(location[i])
                    gaze_data = list(map(lambda q: (float(q[0]), float(q[1]), 1), Video_location))
                    heatmap.draw_heatmap(gaze_data, (monitorWidth, monitorHeight), alpha=0.5,
                                         savefilename='./heatmap/' + str(i) + '.png',
                                         imagefile='frame/' + str(i) + '.png', gaussianwh=200)
                    if len(Video_location) == 30:
                        del Video_location[0]
                Video = False
                Cap = True
            # cv.imwrite('frame/' + str(count) + '.png', video_frame)
        # 비디오의 한 프레임씩 읽습니다. 제대로 프레임을 읽으면 ret값이 True, 실패하면 False가 나타납니다. frame에 읽은 프레임이 나옵니다
        if not ret:
            break
        # 도형임
        if Show:
            image = np.full((monitorHeight, monitorWidth, 3), (255, 255, 255), np.uint8)
            cv.namedWindow('image', cv.WINDOW_NORMAL)
            cv.setWindowProperty('image', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        # ------------------------
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # 색상
        img_h, img_w = frame.shape[:2]  # 모든 랜드마크는 x, y 및 z 값을 가지며 각각의 값은 0에서 1 사이 , 즉 정규화된 값입니다.
        results = face_mesh.process(rgb_frame)

        right_eye = []
        left_eye = []
        face_3d = []
        face_2d = []

        try:
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                            elif idx == 263:
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                left_eye.append([x, y])
                            elif idx == 33:
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                right_eye.append([x, y])

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            # 2D 좌표 얻기
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            # 3D 좌표 얻기
                            face_3d.append([x, y, lm.z])

                        if idx == 159:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            right_eye.append([x, y])
                        elif idx == 133:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            right_eye.append([x, y])
                        elif idx == 145:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            right_eye.append([x, y])
                        elif idx == 386:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            left_eye.append([x, y])
                        elif idx == 362:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            left_eye.append([x, y])
                        elif idx == 374:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            left_eye.append([x, y])

                # region 동공
                left_eye_x_dis = left_eye[0][0]
                left_eye_y_dis = left_eye[0][1]
                left_eye_dis = (left_eye_x_dis, left_eye_y_dis)

                right_eye_x_dis = right_eye[0][0]
                right_eye_y_dis = right_eye[0][1]
                right_eye_dis = (right_eye_x_dis, right_eye_y_dis)

                # eye-eye distance
                eye_length = findDistance(left_eye_dis, right_eye_dis)
                real_eye_dis = 6.3  # 실제 거리
                eye_f = 840  # focal length
                eye_dis = (real_eye_dis * eye_f) / eye_length

                # 동공 영역 추출
                left_eye_x_roi = int((left_eye[0][0] + left_eye[1][0] + left_eye[2][0] + left_eye[3][0]) / 4)
                left_eye_y_roi = int((left_eye[0][1] + left_eye[1][1] + left_eye[2][1] + left_eye[3][1]) / 4)
                left_eye_roi = (left_eye_x_roi, left_eye_y_roi)

                right_eye_x_roi = int((right_eye[0][0] + right_eye[1][0] + right_eye[2][0] + right_eye[3][0]) / 4)
                right_eye_y_roi = int((right_eye[0][1] + right_eye[1][1] + right_eye[2][1] + right_eye[3][1]) / 4)
                right_eye_roi = (right_eye_x_roi, right_eye_y_roi)

                left_pupil = frame[left_eye_y_roi - 24: left_eye_y_roi + 24, left_eye_x_roi - 24: left_eye_x_roi + 24]
                right_pupil = frame[right_eye_y_roi - 24: right_eye_y_roi + 24,
                              right_eye_x_roi - 24: right_eye_x_roi + 24]

                # 왼쪽 동공 좌표
                left_pupil = left_pupil.astype('float32') / 255.  # 정규화
                left_pupil_contour = pupil_model.predict(np.expand_dims(left_pupil, axis=0)).reshape((-1, 2))
                left_pupil_center_x = 0
                left_pupil_center_y = 0
                for i in range(len(left_pupil_contour)):
                    x = int(left_pupil_contour[i][0] + left_eye_x_roi - 24)
                    y = int(left_pupil_contour[i][1] + left_eye_y_roi - 24)
                    p = (x, y)
                    # lineType : 선 타입. cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA 중 선택
                    cv.circle(frame, p, radius=int(1), color=(0, 0, 1),
                              thickness=-1, lineType=cv.LINE_AA)

                # 왼쪽 동공 중심 구하기
                for i in range(len(left_pupil_contour)):
                    left_pupil_center_x += left_pupil_contour[i][0]
                    left_pupil_center_y += left_pupil_contour[i][1]
                left_pupil_center = (int(left_pupil_center_x / 8) + left_eye_x_roi - 24,
                                     int(left_pupil_center_y / 8) + left_eye_y_roi - 24)

                cv.circle(frame, left_pupil_center, radius=int(1), color=(0, 0, 255), thickness=2,
                          lineType=cv.LINE_AA)
                cv.circle(frame, left_eye_roi, radius=int(1), color=(255, 0, 0), thickness=2, lineType=cv.LINE_AA)

                # 오른쪽 동공 좌표
                right_pupil = right_pupil.astype('float32') / 255.  # 정규화
                right_pupil_contour = pupil_model.predict(np.expand_dims(right_pupil, axis=0)).reshape((-1, 2))
                right_pupil_center_x = 0
                right_pupil_center_y = 0
                for i in range(len(right_pupil_contour)):
                    x = int(right_pupil_contour[i][0] + right_eye_x_roi - 24)
                    y = int(right_pupil_contour[i][1] + right_eye_y_roi - 24)
                    p = (x, y)
                    # lineType : 선 타입. cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA 중 선택
                    cv.circle(frame, p, radius=int(1), color=(0, 0, 1),
                              thickness=-1, lineType=cv.LINE_AA)

                # 오른쪽 동공 중심 구하기
                for i in range(len(right_pupil_contour)):
                    right_pupil_center_x += right_pupil_contour[i][0]
                    right_pupil_center_y += right_pupil_contour[i][1]
                right_pupil_center = (int(right_pupil_center_x / 8) + right_eye_x_roi - 24,
                                      int(right_pupil_center_y / 8) + right_eye_y_roi - 24)

                cv.circle(frame, right_pupil_center, radius=int(1), color=(0, 0, 255), thickness=2, lineType=cv.LINE_AA)
                cv.circle(frame, right_eye_roi, radius=int(1), color=(255, 0, 0), thickness=2, lineType=cv.LINE_AA)

                # 왼쪽 동공 이동량
                left_pupil_move_x = left_eye[2][0] - left_pupil_center[0]
                left_pupil_move_y = left_eye[2][1] - left_pupil_center[1]

                # 오른쪽 동공 이동량
                right_pupil_move_x = right_eye[2][0] - right_pupil_center[0]
                right_pupil_move_y = right_eye[2][1] - right_pupil_center[1]

                # endregion

                # region 3D
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

                # Get the y rotation degree
                xa = angles[0] * 360
                ya = angles[1] * 360
                za = angles[2] * 360

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + ya * 10), int(nose_2d[1] - xa * 10))

                cv.line(frame, p1, p2, (255, 0, 0), 3)
                # endregion

                # region save
                move_dis = 30
                if Dot is True:
                    pt = np.array([dot_x, dot_y], dtype=np.int32)
                    cv.circle(image, pt, 30, (0, 0, 255), -1, cv.LINE_AA)

                    if Center is True:
                        if monitorWidth / 2 - move_dis < dot_x < monitorWidth / 2 + move_dis:
                            center_location.append(left_pupil_center)
                            dot_x += move_dis
                        elif dot_x < monitorWidth - 50:
                            dot_x += move_dis
                        else:
                            Center = False
                            Left = True

                    elif Left is True:
                        if monitorHeight / 2 - move_dis < dot_y < monitorHeight / 2 + move_dis:
                            left_location.append(left_pupil_center)
                            dot_y -= move_dis
                        elif dot_y > 50:
                            dot_y -= move_dis
                        else:
                            Left = False
                            LeftUp = True

                    elif LeftUp is True:
                        if monitorWidth - 50 - move_dis < dot_x < monitorWidth - 50 + move_dis:
                            leftup_location.append(left_pupil_center)
                            dot_x -= move_dis
                        elif dot_x > monitorWidth / 2:
                            dot_x -= move_dis
                        else:
                            LeftUp = False
                            CenterUp = True

                    elif CenterUp is True:
                        if monitorWidth / 2 - move_dis <= dot_x <= monitorWidth / 2 + move_dis:
                            centerup_location.append(left_pupil_center)
                            dot_x -= move_dis
                        elif dot_x > 50:
                            dot_x -= move_dis
                        else:
                            CenterUp = False
                            RightUp = True

                    elif RightUp is True:
                        if 50 - move_dis < dot_y < 50 + move_dis:
                            rightup_location.append(left_pupil_center)
                            dot_y += move_dis
                        elif dot_y < monitorHeight / 2:
                            dot_y += move_dis
                        else:
                            RightUp = False
                            Right = True

                    elif Right is True:
                        if monitorHeight / 2 - move_dis < dot_y < monitorHeight / 2 + move_dis:
                            right_location.append(left_pupil_center)
                            dot_y += move_dis
                        elif dot_y < monitorHeight - 50:
                            dot_y += move_dis
                        else:
                            Right = False
                            RightDown = True

                    elif RightDown is True:
                        if 50 - move_dis < dot_x < 50 + move_dis:
                            rightdown_location.append(left_pupil_center)
                            dot_x += move_dis
                        elif dot_x < monitorWidth / 2:
                            dot_x += move_dis
                        else:
                            RightDown = False
                            CenterDown = True

                    elif CenterDown is True:
                        if monitorWidth / 2 - move_dis <= dot_x <= monitorWidth / 2 + move_dis:
                            centerdown_location.append(left_pupil_center)
                            dot_x += move_dis
                        elif dot_x < monitorWidth - 50:
                            dot_x += move_dis
                        else:
                            CenterDown = False
                            LeftDown = True

                    elif LeftDown is True:
                        leftdown_location.append(left_pupil_center)
                        LeftDown = False
                        MoveCenter = True

                    elif MoveCenter:
                        if dot_x > monitorWidth / 2:
                            dot_x -= monitorWidth / 20
                        if dot_y > monitorHeight / 2:
                            dot_y -= monitorHeight / 20
                        else:
                            MoveCenter = False
                            Dot = False
                            Center_nose = True


                # 파란점
                elif Center_nose is True:
                    pt = np.array([dot_nose_x, dot_nose_y], dtype=np.int32)
                    cv.circle(image, pt, 30, (255, 0, 0), -1, cv.LINE_AA)
                    if monitorWidth / 2 - move_dis < dot_nose_x < monitorWidth / 2 + move_dis:
                        center_location_nose.append(left_pupil_center)
                        centerxa = xa
                        centerya = ya
                        dot_nose_x += move_dis
                    elif dot_nose_x < monitorWidth - 50:
                        dot_nose_x += move_dis
                    else:
                        Center_nose = False
                        Left_nose = True

                elif Left_nose is True:
                    pt = np.array([dot_nose_x, dot_nose_y], dtype=np.int32)
                    cv.circle(image, pt, 30, (255, 0, 0), -1, cv.LINE_AA)
                    if monitorHeight / 2 - move_dis < dot_nose_y < monitorHeight / 2 + move_dis:
                        left_location_nose.append(left_pupil_center)
                        leftxa = xa
                        leftya = ya
                        dot_nose_y -= move_dis
                    elif dot_nose_y > 50:
                        dot_nose_y -= move_dis
                    else:
                        Left_nose = False
                        LeftUp_nose = True

                elif LeftUp_nose is True:
                    pt = np.array([dot_nose_x, dot_nose_y], dtype=np.int32)
                    cv.circle(image, pt, 30, (255, 0, 0), -1, cv.LINE_AA)
                    if monitorWidth - 50 - move_dis < dot_nose_x < monitorWidth - 50 + move_dis:
                        leftup_location_nose.append(left_pupil_center)
                        dot_nose_x -= move_dis
                    elif dot_nose_x > monitorWidth / 2:
                        dot_nose_x -= move_dis
                    else:
                        LeftUp_nose = False
                        CenterUp_nose = True

                elif CenterUp_nose is True:
                    pt = np.array([dot_nose_x, dot_nose_y], dtype=np.int32)
                    cv.circle(image, pt, 30, (255, 0, 0), -1, cv.LINE_AA)
                    if monitorWidth / 2 - move_dis <= dot_nose_x <= monitorWidth / 2 + move_dis:
                        centerup_location_nose.append(left_pupil_center)
                        upxa = xa
                        upya = ya
                        dot_nose_x -= move_dis
                    elif dot_nose_x > 50:
                        dot_nose_x -= move_dis
                    else:
                        CenterUp_nose = False
                        RightUp_nose = True

                elif RightUp_nose is True:
                    pt = np.array([dot_nose_x, dot_nose_y], dtype=np.int32)
                    cv.circle(image, pt, 30, (255, 0, 0), -1, cv.LINE_AA)
                    if 50 - move_dis < dot_nose_y < 50 + move_dis:
                        rightup_location_nose.append(left_pupil_center)
                        dot_nose_y += move_dis
                    elif dot_nose_y < monitorHeight / 2:
                        dot_nose_y += move_dis
                    else:
                        RightUp_nose = False
                        Right_nose = True

                elif Right_nose is True:
                    pt = np.array([dot_nose_x, dot_nose_y], dtype=np.int32)
                    cv.circle(image, pt, 30, (255, 0, 0), -1, cv.LINE_AA)
                    if monitorHeight / 2 - move_dis < dot_nose_y < monitorHeight / 2 + move_dis:
                        right_location_nose.append(left_pupil_center)
                        rightxa = xa
                        rightya = ya
                        dot_nose_y += move_dis
                    elif dot_nose_y < monitorHeight - 50:
                        dot_nose_y += move_dis
                    else:
                        Right_nose = False
                        RightDown_nose = True

                elif RightDown_nose is True:
                    pt = np.array([dot_nose_x, dot_nose_y], dtype=np.int32)
                    cv.circle(image, pt, 30, (255, 0, 0), -1, cv.LINE_AA)
                    if 50 - move_dis < dot_nose_x < 50 + move_dis:
                        rightdown_location_nose.append(left_pupil_center)
                        dot_nose_x += move_dis
                    elif dot_nose_x < monitorWidth / 2:
                        dot_nose_x += move_dis
                    else:
                        RightDown_nose = False
                        CenterDown_nose = True

                elif CenterDown_nose is True:
                    pt = np.array([dot_nose_x, dot_nose_y], dtype=np.int32)
                    cv.circle(image, pt, 30, (255, 0, 0), -1, cv.LINE_AA)
                    if monitorWidth / 2 - move_dis <= dot_nose_x <= monitorWidth / 2 + move_dis:
                        centerdown_location_nose.append(left_pupil_center)
                        downxa = xa
                        downya = ya
                        dot_nose_x += move_dis
                    elif dot_nose_x < monitorWidth - 50:
                        dot_nose_x += move_dis
                    else:
                        CenterDown_nose = False
                        LeftDown_nose = True

                elif LeftDown_nose is True:
                    pt = np.array([dot_nose_x, dot_nose_y], dtype=np.int32)
                    cv.circle(image, pt, 30, (255, 0, 0), -1, cv.LINE_AA)
                    leftdown_location_nose.append(left_pupil_center)
                    LeftDown_nose = False
                    Start = True
                    Show = False
                    Circle = True
                    cv.destroyAllWindows()

                # endregion
                elif Start:
                    for i in range(len(center_location)):
                        if i == 0:
                            center_pupil_x = center_location[i][0]
                            center_pupil_y = center_location[i][1]
                            center_pupil = (center_pupil_x, center_pupil_y)
                        else:
                            center_pupil_x = (center_pupil_x + center_location[i][0]) / 2
                            center_pupil_y = (center_pupil_y + center_location[i][1]) / 2
                            center_pupil = (center_pupil_x, center_pupil_y)

                    for i in range(len(rightup_location)):
                        if i == 0:
                            rightup_pupil_x = rightup_location[i][0]
                            rightup_pupil_y = rightup_location[i][1]
                            rightup_pupil = (rightup_pupil_x, rightup_pupil_y)
                        else:
                            rightup_pupil_x = (rightup_pupil_x + rightup_location[i][0]) / 2
                            rightup_pupil_y = (rightup_pupil_y + rightup_location[i][1]) / 2
                            rightup_pupil = (rightup_pupil_x, rightup_pupil_y)

                    for i in range(len(rightdown_location)):
                        if i == 0:
                            rightdown_pupil_x = rightdown_location[i][0]
                            rightdown_pupil_y = rightdown_location[i][1]
                            rightdown_pupil = (rightdown_pupil_x, rightdown_pupil_y)
                        else:
                            rightdown_pupil_x = (rightdown_pupil_x + rightdown_location[i][0]) / 2
                            rightdown_pupil_y = (rightdown_pupil_y + rightdown_location[i][1]) / 2
                            rightdown_pupil = (rightdown_pupil_x, rightdown_pupil_y)

                    for i in range(len(leftup_location)):
                        if i == 0:
                            leftup_pupil_x = leftup_location[i][0]
                            leftup_pupil_y = leftup_location[i][1]
                            leftup_pupil = (leftup_pupil_x, leftup_pupil_y)
                        else:
                            leftup_pupil_x = (leftup_pupil_x + leftup_location[i][0]) / 2
                            leftup_pupil_y = (leftup_pupil_y + leftup_location[i][1]) / 2
                            leftup_pupil = (leftup_pupil_x, leftup_pupil_y)

                    for i in range(len(leftdown_location)):
                        if i == 0:
                            leftdown_pupil_x = leftdown_location[i][0]
                            leftdown_pupil_y = leftdown_location[i][1]
                            leftdown_pupil = (leftdown_pupil_x, leftdown_pupil_y)
                        else:
                            leftdown_pupil_x = (leftdown_pupil_x + leftdown_location[i][0]) / 2
                            leftdown_pupil_y = (leftdown_pupil_y + leftdown_location[i][1]) / 2
                            leftdown_pupil = (leftdown_pupil_x, leftdown_pupil_y)

                    for i in range(len(centerup_location)):
                        if i == 0:
                            centerup_pupil_x = centerup_location[i][0]
                            centerup_pupil_y = centerup_location[i][1]
                            centerup_pupil = (centerup_pupil_x, centerup_pupil_y)
                        else:
                            centerup_pupil_x = (centerup_pupil_x + centerup_location[i][0]) / 2
                            centerup_pupil_y = (centerup_pupil_y + centerup_location[i][1]) / 2
                            centerup_pupil = (centerup_pupil_x, centerup_pupil_y)

                    for i in range(len(centerdown_location)):
                        if i == 0:
                            centerdown_pupil_x = centerdown_location[i][0]
                            centerdown_pupil_y = centerdown_location[i][1]
                            centerdown_pupil = (centerdown_pupil_x, centerdown_pupil_y)
                        else:
                            centerdown_pupil_x = (centerdown_pupil_x + centerdown_location[i][0]) / 2
                            centerdown_pupil_y = (centerdown_pupil_y + centerdown_location[i][1]) / 2
                            centerdown_pupil = (centerdown_pupil_x, centerdown_pupil_y)

                    for i in range(len(left_location)):
                        if i == 0:
                            left_pupil1_x = left_location[i][0]
                            left_pupil1_y = left_location[i][1]
                            left_pupil1 = (left_pupil1_x, left_pupil1_y)
                        else:
                            left_pupil1_x = (left_pupil1_x + left_location[i][0]) / 2
                            left_pupil1_y = (left_pupil1_y + left_location[i][1]) / 2
                            left_pupil1 = (left_pupil1_x, left_pupil1_y)

                    for i in range(len(right_location)):
                        if i == 0:
                            right_pupil1_x = right_location[i][0]
                            right_pupil1_y = right_location[i][1]
                            right_pupil1 = (right_pupil1_x, right_pupil1_y)
                        else:
                            right_pupil1_x = (right_pupil1_x + right_location[i][0]) / 2
                            right_pupil1_y = (right_pupil1_y + right_location[i][1]) / 2
                            right_pupil1 = (right_pupil1_x, right_pupil1_y)

                    for i in range(len(center_location_nose)):
                        if i == 0:
                            center_x = center_location_nose[i][0]
                            center_y = center_location_nose[i][1]
                            center = (center_x, center_y)
                        else:
                            center_x = (center_x + center_location_nose[i][0]) / 2
                            center_y = (center_y + center_location_nose[i][1]) / 2
                            center = (center_x, center_y)

                    for i in range(len(rightup_location_nose)):
                        if i == 0:
                            rightup_x = rightup_location_nose[i][0]
                            rightup_y = rightup_location_nose[i][1]
                            rightup = (rightup_x, rightup_y)
                        else:
                            rightup_x = (rightup_x + rightup_location_nose[i][0]) / 2
                            rightup_y = (rightup_y + rightup_location_nose[i][1]) / 2
                            rightup = (rightup_x, rightup_y)

                    for i in range(len(rightdown_location_nose)):
                        if i == 0:
                            rightdown_x = rightdown_location_nose[i][0]
                            rightdown_y = rightdown_location_nose[i][1]
                            rightdown = (rightdown_x, rightdown_y)
                        else:
                            rightdown_x = (rightdown_x + rightdown_location_nose[i][0]) / 2
                            rightdown_y = (rightdown_y + rightdown_location_nose[i][1]) / 2
                            rightdown = (rightdown_x, rightdown_y)

                    for i in range(len(leftup_location_nose)):
                        if i == 0:
                            leftup_x = leftup_location_nose[i][0]
                            leftup_y = leftup_location_nose[i][1]
                            leftup = (leftup_x, leftup_y)
                        else:
                            leftup_x = (leftup_x + leftup_location_nose[i][0]) / 2
                            leftup_y = (leftup_y + leftup_location_nose[i][1]) / 2
                            leftup = (leftup_x, leftup_y)

                    for i in range(len(leftdown_location_nose)):
                        if i == 0:
                            leftdown_x = leftdown_location_nose[i][0]
                            leftdown_y = leftdown_location_nose[i][1]
                            leftdown = (leftdown_x, leftdown_y)
                        else:
                            leftdown_x = (leftdown_x + leftdown_location_nose[i][0]) / 2
                            leftdown_y = (leftdown_y + leftdown_location_nose[i][1]) / 2
                            leftdown = (leftdown_x, leftdown_y)

                    for i in range(len(centerup_location_nose)):
                        if i == 0:
                            centerup_x = centerup_location_nose[i][0]
                            centerup_y = centerup_location_nose[i][1]
                            centerup = (centerup_x, centerup_y)
                        else:
                            centerup_x = (centerup_x + centerup_location_nose[i][0]) / 2
                            centerup_y = (centerup_y + centerup_location_nose[i][1]) / 2
                            centerup = (centerup_x, centerup_y)

                    for i in range(len(centerdown_location_nose)):
                        if i == 0:
                            centerdown_x = centerdown_location_nose[i][0]
                            centerdown_y = centerdown_location_nose[i][1]
                            centerdown = (centerdown_x, centerdown_y)
                        else:
                            centerdown_x = (centerdown_x + centerdown_location_nose[i][0]) / 2
                            centerdown_y = (centerdown_y + centerdown_location_nose[i][1]) / 2
                            centerdown = (centerdown_x, centerdown_y)

                    for i in range(len(left_location_nose)):
                        if i == 0:
                            left_x = left_location_nose[i][0]
                            left_y = left_location_nose[i][1]
                            left = (left_x, left_y)
                        else:
                            left_x = (left_x + left_location_nose[i][0]) / 2
                            left_y = (left_y + left_location_nose[i][1]) / 2
                            left = (left_x, left_y)

                    for i in range(len(right_location_nose)):
                        if i == 0:
                            right_x = right_location_nose[i][0]
                            right_y = right_location_nose[i][1]
                            right = (right_x, right_y)
                        else:
                            right_x = (right_x + right_location_nose[i][0]) / 2
                            right_y = (right_y + right_location_nose[i][1]) / 2
                            right = (right_x, right_y)

                    Start = False
                    Eye_Tracking = True
                    Cap = True

                elif Eye_Tracking:
                    xs = left[0] - right[0]
                    ys = centerdown[1] - centerup[1]

                    xc = left_pupil1[0] - right_pupil1[0]
                    yc = centerdown_pupil[1] - centerup_pupil[1]

                    xd = (xs + xc) / 2
                    yd = (ys + yc) / 2

                    min_x = xd / 100
                    min_y = yd / 100

                    move_x = left_pupil_center[0] - rightup[0]
                    move_y = left_pupil_center[1] - rightup[1]

                    per_x = move_x / min_x  # x = **%
                    per_y = move_y / min_y  # y = **%

                    real_x = int(monitorWidth * (per_x / 100))
                    real_y = int(monitorHeight * (per_y / 100))

                    # 좌표값 저장
                    location_x = monitorWidth - real_x
                    location_y = real_y
                    location.append([location_x, location_y])

                    print(monitorWidth - real_x)
                    print(real_y)

                    if Walk:
                        s.move((location[-1][0], location[-1][1]))

                    if TwoImage:
                        if Clock:
                            twoimage = cv.imread("clock.png")
                            if len(location) <= 100:
                                if location[-1][0] < monitorWidth / 2:
                                    if location[-1][1] > monitorHeight / 2:
                                        left_down_clock += 1
                                    else:
                                        left_top_clock += 1
                                elif location[-1][0] >= monitorWidth / 2:
                                    if location[-1][1] > monitorHeight / 2:
                                        right_down_clock += 1
                                    else:
                                        right_top_clock += 1

                            else:
                                cv.destroyWindow('TwoImage')
                                Clock = False
                                Clock_HeatMap = True

                    if Clock_HeatMap:
                        h, w, _ = twoimage.shape
                        xx = ['left_top_clock', 'left_down_clock', 'right_top_clock', 'right_down_clock']
                        yy = [left_top_clock, left_down_clock, right_top_clock, right_down_clock]
                        plt.bar(xx, yy, color=['r', 'g', 'b', 'y'], alpha=0.4)
                        plt.ylabel("%")
                        plt.savefig('graph.png')
                        for i in range(len(location)):
                            clock_x = location[i][0] / monitorWidth * w
                            clock_y = location[i][1] / monitorHeight * h
                            Clock_location.append((clock_x, clock_y))
                        gaze_data = list(map(lambda q: (float(q[0]), float(q[1]), 1), Clock_location))
                        heatmap.draw_heatmap(gaze_data, (w, h), alpha=0.5,
                                             savefilename='result1.png',
                                             imagefile='clock.png', gaussianwh=200)
                        plt.show(block=False)
                        TwoImage = False
                        Clock_HeatMap = False
                        Cap = True
                        location.clear()

                    # 마우스 제어
                    # pyautogui.moveTo(monitorWidth - real_x, real_y, 0.1)
        except:
            print('예외발생1')

        frame = cv.flip(frame, 1)
        # 얼굴 사이즈 검은색으로 만듬
        pts = np.array([[1, 1], [1, 480], [130, 480], [170, 340], [250, 300], [230, 150], [300, 80],
                        [340, 80], [410, 150], [390, 300], [470, 340], [510, 480], [640, 480], [640, 1], [1, 1]])
        cv.fillPoly(frame, [pts], (0, 0, 0), cv.LINE_AA)
        try:
            cv.putText(frame, 'depth : ' + str(int(eye_dis)), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        except:
            print('예외발생2')
        if Cap:
            cv.imshow('frame', frame)

        if Show:
            image = cv.flip(image, 1)
            text1 = np.array([monitorWidth / 11, 200], dtype=np.int32)
            text2 = np.array([monitorWidth / 11, 280], dtype=np.int32)
            text3 = np.array([monitorWidth / 11, 360], dtype=np.int32)
            text4 = np.array([monitorWidth - monitorWidth / 3, 200], dtype=np.int32)
            text5 = np.array([monitorWidth - monitorWidth / 3, 280], dtype=np.int32)
            try:
                cv.putText(image, 'Dot : ' + str(pt), text1, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                cv.putText(image, 'Left_Eye : (' + str(int(left_pupil_center[0])) + ', ' + str(
                    int(left_pupil_center[1])) + ')', text2, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                cv.putText(image, 'Right_Eye : (' + str(int(right_pupil_center[0])) + ', ' + str(
                    int(right_pupil_center[1])) + ')', text3, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                cv.putText(image, 'column : ' + str(int(xa)), text4, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                cv.putText(image, 'row : ' + str(int(ya)), text5, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            except:
                print('예외발생3')
            cv.imshow('image', image)

        elif Video:
            current_time = time.time() - prev_time
            if (suc is True) and (current_time > 1. / fps):
                prev_time = time.time()
                cv.imwrite('frame/' + str(count) + '.png', video_frame)

            cv.namedWindow('video_frame', cv.WINDOW_NORMAL)
            cv.setWindowProperty('video_frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            cv.imshow('video_frame', video_frame)
            count += 1

        elif TwoImage:
            cv.namedWindow('TwoImage', cv.WINDOW_NORMAL)
            cv.setWindowProperty('TwoImage', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            cv.imshow('TwoImage', twoimage)

        if Circle:
            s = Sticker.Sticker('TWOMYUNG.png', xy=[-50, -50], size=0.3, on_top=True)
            Circle = False
            Walk = True

        key = cv.waitKey(1)
        if key == 27:
            break

        elif key == ord('s'):
            Cap = False
            Dot = True
            Center = True
            cv.destroyWindow("frame")

        elif key == ord('c'):
            location.clear()
            TwoImage = True
            Cap = False
            Clock = True
            cv.destroyWindow("frame")

        elif key == ord('o'):
            location.clear()
            Video = True
            Cap = False
            cv.destroyWindow("frame")

        elif key == ord('r'):
            Init = True

cap.release()
cv.destroyAllWindows()
sys.exit()
# sys.exit(app.exec_())
