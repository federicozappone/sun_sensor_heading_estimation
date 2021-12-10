import numpy as np
import datetime
import time
import math
import cv2
import yaml

from scipy.spatial.transform import Rotation
from image_utils import image_resize


def spherical_to_cartesian(r, elevation, azimuth):
    x = r * math.sin(elevation) * math.cos(azimuth)
    y = r * math.sin(elevation) * math.sin(azimuth)
    z = r * math.cos(elevation)
    return x, y, z


def into_range(x, range_min, range_max):
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min


def sun_position(when, location, refraction):
    year, month, day, hour, minute, second, timezone = when
    latitude, longitude = location
    # convert latitude and longitude to radians
    rlat = math.radians(latitude)
    rlon = math.radians(longitude)
    # decimal hour of the day at Greenwich
    greenwichtime = hour - timezone + minute / 60 + second / 3600
    # days from J2000, accurate from 1901 to 2099
    daynum = (
        367 * year
        - 7 * (year + (month + 9) // 12) // 4
        + 275 * month // 9
        + day
        - 730531.5
        + greenwichtime / 24
    )
    # mean longitude of the sun
    mean_long = daynum * 0.01720279239 + 4.894967873
    # mean anomaly of the Sun
    mean_anom = daynum * 0.01720197034 + 6.240040768
    # ecliptic longitude of the sun
    eclip_long = (
        mean_long
        + 0.03342305518 * math.sin(mean_anom)
        + 0.0003490658504 * math.sin(2 * mean_anom)
    )
    # obliquity of the ecliptic
    obliquity = 0.4090877234 - 0.000000006981317008 * daynum
    # right ascension of the sun
    rasc = math.atan2(math.cos(obliquity) * math.sin(eclip_long), math.cos(eclip_long))
    # declination of the sun
    decl = math.asin(math.sin(obliquity) * math.sin(eclip_long))
    # local sidereal time
    sidereal = 4.894961213 + 6.300388099 * daynum + rlon
    # hour angle of the sun
    hour_ang = sidereal - rasc
    # local elevation of the sun
    elevation = math.asin(math.sin(decl) * math.sin(rlat) + math.cos(decl) * math.cos(rlat) * math.cos(hour_ang))
    # local azimuth of the sun
    azimuth = math.atan2(
        -math.cos(decl) * math.cos(rlat) * math.sin(hour_ang),
        math.sin(decl) - math.sin(rlat) * math.sin(elevation),
    )
    # convert azimuth and elevation to degrees
    azimuth = into_range(math.degrees(azimuth), 0, 360)
    elevation = into_range(math.degrees(elevation), -180, 180)
    # refraction correction (optional)
    if refraction:
        targ = math.radians((elevation + (10.3 / (elevation + 5.11))))
        elevation += (1.02 / math.tan(targ)) / 60

    # compute sun distance for a given day of the year
    day_of_the_year = datetime.datetime.now().timetuple().tm_yday # day from january 1
    au = 149597870700 # meters
    distance = 1.0 - 0.01672 * math.cos(((2 * math.pi) / 365.256363) * (day_of_the_year - 4))
    distance *= au

    # return azimuth and elevation in degrees
    return (round(distance, 2), round(azimuth, 2), round(elevation, 2))


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix


# preliminary detection method, still not properly tested
def detect_sun_position(frame, camera_matrix, dist_coefs):
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

    dst = cv2.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)

    # crop the frame
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (41, 41), 0)

    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(gray)

    return max_loc


if __name__ == "__main__":

    parameters = yaml.load(open("configs/config.yaml", "r"), Loader=yaml.FullLoader)

    # acquire frames from the system camera with id 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("cannot open camera")
        exit()

    location = (parameters["location"]["lat"], parameters["location"]["lon"]) # lat, lon

    try:
        with np.load("calibration_data/calibration.npz") as X:
            camera_matrix, dist_coefs, \
            rvecs, tvecs, w, h, pattern_size, rms = [X[i] for i in ("camera_matrix", "dist_coefs", "rvecs", 
                                                                    "tvecs", "w", "h", "pattern_size", "rms")]
    except FileNotFoundError:
        print("couldn't find calibration data")
        exit()


    while True:
        ret, frame = cap.read()

        if ret is False or frame is None:
            print("could not acquire frame")
            continue

        dt = datetime.datetime.now()

        offset = time.timezone if (time.localtime().tm_isdst == 0) else time.altzone
        offset = offset / 60 / 60 * -1

        when = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, offset)
        # get the Sun's apparent location in the sky
        distance, azimuth, elevation = sun_position(when, location, True)
        # output the results
        print("\ndate time:", when)
        print("location:", location)

        print("\nsun spherical coordinates")
        print("distance:", distance)
        print("azimuth:", azimuth)
        print("elevation:", elevation)

        x, y, z = spherical_to_cartesian(distance, math.radians(elevation), math.radians(azimuth))

        z = abs(z)

        print("\nsun cartesian coordinates")
        print("x:", x)
        print("y:", y)
        print("z:", z)

        debug_image = np.zeros((h, w, 3), dtype=np.uint8)

        I = np.eye(3)
        t = np.zeros(3)
        R = Rotation.from_euler("y", 90, degrees=True) # camera pointing up

        print("\ncamera matrix")
        print(camera_matrix)

        translation_3d = np.c_[I, t]
        rotation_3d = np.c_[np.r_[R.as_matrix(), [np.zeros(3)]], [0, 0, 0, 1]]

        print("\ncamera 3d translation")
        print(translation_3d)
        print("camera 3d rotation")
        print(rotation_3d)

        P = np.array([x, y, z, 1]).T

        # assuming undistorted camera
        p = camera_matrix @ translation_3d @ rotation_3d @ P

        print("\nprojected sun position in image plane:", p)

        p /= p[2]

        print("\nprojected sun position in image plane (normalized):", p)

        u_est = int(p[0])
        v_est = int(p[1])


        if u_est > w or v_est > h or u_est < 0 or v_est < 0:
            print("sun projection outside of image plane")
            continue


        u_real, v_real = detect_sun_position(frame, camera_matrix, dist_coefs)

        # plot estimated sun position on debug image
        cv2.circle(debug_image, (u_est, v_est), 4, (255, 255, 255), -1)
        cv2.line(debug_image, (w // 2, h // 2), (u_est, v_est), (255, 255, 255), 1)

        # plot detected sun position on debug image
        cv2.circle(debug_image, (u_real, v_real), 4, (0, 255, 0), -1)
        cv2.line(debug_image, (w // 2, h // 2), (u_real, v_real), (0, 255, 0), 1)

        # plot detected sun position on the camera frame
        cv2.circle(frame, (u_real, v_real), 8, (255, 0, 0), 1)

        # compute rotation matrix from one vector to another
        rot = rotation_matrix_from_vectors([u_est, v_est, 0], [u_real, v_real, 0])
        
        rot = Rotation.from_matrix(rot)

        print("\ncamera estimated rotation matrix")
        print(rot.as_matrix())

        print("camera estimated rotation vector")
        print(rot.as_rotvec())

        camera_heading = azimuth + math.degrees(rot.as_rotvec()[2])

        print("\ncamera heading (deg):", camera_heading)


        cv2.imshow("camera frame", frame)
        cv2.imshow("debug image", debug_image)
        cv2.waitKey(100)
