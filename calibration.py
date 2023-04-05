import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt


# 코너의 가로, 세로, 사이즈, 불러올 이미지 정하기
checkboard_x = 6
checkboard_y = 8
checkboard_size = 30
folder_path = './image'                         # 이미지들이 들어있는 폴더 경로
image_list = glob.glob(folder_path + '/*.jpg')  # 확장자명에 따라 변경하기

# 정확도 향상, 체크보드 그리드 포인트 생성
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
obj_points = []     # 실제 3D 포인트
img_points = []     # 실제 2D 포인트
obj_grid = np.zeros((1, checkboard_x*checkboard_y, 3), np.float32)
obj_grid[0, :, :2] = np.mgrid[0:6,0:8].T.reshape(-1, 2)



#이미지 가져오기
for image_path in image_list:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이미지에서 체크보드 패턴 찾기
    ret, corners = cv2.findChessboardCorners(gray,
                                                (checkboard_x,checkboard_y),
                                                cv2.CALIB_CB_ADAPTIVE_THRESH 
                                            + cv2.CALIB_CB_FAST_CHECK 
                                            + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        obj_grid = obj_grid * checkboard_size
        obj_points.append(obj_grid)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)    # 코너 좌표 정확도 높이기
        img_points.append(corners2)
        img = cv2.drawChessboardCorners(img, (6,8), corners2, ret)

# calibration 실행
ret, camera_matrix, dist_coef, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# undistort을 보여줄 새로운 매트릭스 정의
h, w = img.shape[:2]
new_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coef, (w, h), 1, (w, h))

# undistort 하기
img_undst = cv2.undistort(img,camera_matrix, dist_coef, None, new_matrix)

# 카메라 보정 오차 계산
mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coef)
    error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)    
    mean_error += error

# calibration 결과 출력
print("Camera Intrinsic matrix:\n", camera_matrix, end = "\n\n")
print("Distortion coefficients:\n", dist_coef, end= "\n\n")
print("Total Error\n", mean_error / len(obj_points), end="\n\n")
# 이미지에서 검출된 대상이 카메라를 중심으로 어떤 각도로 회전되었는지를 나타내는 벡터. 각각 x, y, z 축을 중심으로 회전하는 각도
print("rvecs:\n", rvecs, end= "\n\n")   
# 월드 좌표계에서 좌표계에서 카메라의 위치를 나타냄
print("tvecs:\n", tvecs, end= "\n\n") 


# 결과 사진 출력
fig, axis = plt.subplots(1, 2, figsize=(10, 5))
axis[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axis[1].imshow(cv2.cvtColor(img_undst, cv2.COLOR_BGR2RGB))
axis[0].set_title('Image calibration')
axis[1].set_title('undistorted Image')
plt.show()
