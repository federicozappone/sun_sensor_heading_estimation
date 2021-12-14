import cv2
import numpy as np
import os
import glob

pattern_size = (6, 9) # chessboard size

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, pattern_size[0] * pattern_size[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

img_shape = None

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob("calibration_data/calibration_frames/*.png")

for fname in images:
    img = cv2.imread(fname)
    if img_shape == None:
        img_shape = img.shape[:2]
    else:
        assert img_shape == img.shape[:2], "all images must share the same size."

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

valid = len(objpoints)

K = np.zeros((3, 3))
D = np.zeros((4, 1))

rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(valid)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(valid)]

rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

print("Valid images:", valid)
print("image shape:", img_shape[::-1])

print("K:", K.tolist())
print("D:", D.tolist())

np.savez("calibration_data/calibration.npz", K=K, D=D)
