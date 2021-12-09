import cv2
import numpy as np
import getopt
import os
import sys

from glob import glob


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)

    return img


def process_image(fn):
    print("Processing %s... " % fn)
    img = cv2.imread(fn, 0)
    if img is None:
        print("Failed to load", fn)
        return None

    assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

    if debug_dir:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)
        _path, name, _ext = splitfn(fn)
        outfile = os.path.join(debug_dir, name + "_chess.png")
        cv2.imwrite(outfile, vis)

    if not found:
        print("Chessboard not found")
        return None

    print("           %s... OK" % fn)
    return (corners.reshape(-1, 2), pattern_points)


if __name__ == "__main__":
    args, img_mask = getopt.getopt(sys.argv[1:], "", ["debug=", "square_size=", "threads="])
    args = dict(args)
    args.setdefault("--debug", "calibration_data/debug")
    args.setdefault("--square_size", 0.002)
    args.setdefault("--threads", 4)
    if not img_mask:
        img_mask = "calibration_data/calibration_frames/*.png"  # default
    else:
        img_mask = img_mask[0]

    img_names = glob(img_mask)
    debug_dir = args.get("--debug")
    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)
    square_size = float(args.get("--square_size"))

    pattern_size = (6, 9)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = cv2.imread(img_names[0], 0).shape[:2]  # TODO: use imquery call to retrieve results

    threads_num = int(args.get("--threads"))
    if threads_num <= 1:
        chessboards = [process_image(fn) for fn in img_names]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(process_image, img_names)

    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)

    np.savez("calibration_data/calibration.npz",
             camera_matrix=camera_matrix, dist_coefs=dist_coefs,
             rvecs=rvecs, tvecs=tvecs, w=w, h=h, pattern_size=pattern_size, rms=rms)

    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients:\n", dist_coefs.ravel())

    # undistort the image with the calibration
    print("")

    for fn in img_names if debug_dir else []:
        path, name, ext = splitfn(fn)
        img_found = os.path.join(debug_dir, name + "_chess.png")
        outfile = os.path.join(debug_dir, name + "_undistorted.png")

        img = cv2.imread(img_found)
        if img is None:
            continue

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        print("Undistorted image written to: %s" % outfile)
        cv2.imwrite(outfile, dst)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    for fname in glob("calibration/calibration_data/calibration_frames/*.png"):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 9), None)

        if ret is True:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

            # Find the rotation and translation vectors.
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            img = draw(img, corners, imgpts)

            _path, name, _ext = splitfn(fname)

            outfile = os.path.join(debug_dir, name + "_pnp.png")
            cv2.imwrite(outfile, img)

    cv2.destroyAllWindows()
