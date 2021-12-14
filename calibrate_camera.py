import cv2
import numpy as np
import os
import glob

def compute_rotation_matrix(w, phi, k, degrees=False):
    """
    Computes Rotation matrix from the rotation angle w, phi, k
    related to x, y and z.
    Returns
    -------
    r_matrix: Rotational Matrix M
         [cos(phi)cos(k)   sin(w)sin(phi)cos(k)+cos(w)sin(k)   -cos(w)sin(phi)cos(k)+sin(w)sin(k)]
     M = [-cos(phi)sin(k)  -sin(w)sin(phi)sin(k)+cos(w)cos(k)  cos(w)sin(phi)sin(k)+sin(w)cos(k) ]
         [   sin(phi)              -sin(w)cos(phi)                        cos(w)cos(phi)         ]
    """

    if degrees is True:
        w = math.radians(w)
        phi = math.radians(phi)
        k = math.radians(k)

    # Rotational Matrix M generation
    r_matrix = np.zeros((3, 3))
    r_matrix[0, 0] = math.cos(phi) * math.cos(k)
    r_matrix[0, 1] = math.sin(w) * math.sin(phi) * math.cos(k) + \
        math.cos(w) * math.sin(k)
    r_matrix[0, 2] = - math.cos(w) * math.sin(phi) * math.cos(k) + \
        math.sin(w) * math.sin(k)
    r_matrix[1, 0] = - math.cos(phi) * math.sin(k)
    r_matrix[1, 1] = - math.sin(w) * math.sin(phi) * math.sin(k) + \
        math.cos(w) * math.cos(k)
    r_matrix[1, 2] = math.cos(w) * math.sin(phi) * math.sin(k) + \
        math.sin(w) * math.cos(k)
    r_matrix[2, 0] = math.sin(phi)
    r_matrix[2, 1] = - math.sin(w) * math.cos(phi)
    r_matrix[2, 2] = math.cos(w) * math.cos(phi)

    return r_matrix

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

center = np.zeros(3)
rotation_mat = compute_rotation_matrix(0, 0, 0)

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

C = center
A = rotation_mat[2, :] - C
Hn = rotation_mat[0, :] - C
Vn = rotation_mat[1, :] - C

H = fx * Hn + cx * A
V = fy * Vn + cy * A

np.savez("calibration_data/calibration.npz", C=C, A=A, H=H, V=V, D=D)
