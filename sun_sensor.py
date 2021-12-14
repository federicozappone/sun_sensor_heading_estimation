import numpy as np
import datetime
import time
import math
import cv2
import yaml

from scipy.spatial.transform import Rotation
from image_utils import image_resize


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

    # return azimuth and elevation in degrees
    return (round(distance, 2), round(azimuth, 2))


def rot_matrix_from_imu(ax, ay, az):
    v = math.sqrt(1 - ax**2)

    G = np.matrix([[v, -(ay * ax) / v, -(az * ax) / v], 
                   [0, az / v, -ay / v], 
                   [ax, ay, az]])

    return G

def rot_matrix_from_roll_pitch(roll, pitch):
    T = np.matrix([[math.cos(pitch), 0, -math.sin(pitch)], 
                   [math.sin(roll) * math.sin(pitch), math.cos(roll), math.sin(roll) * math.cos(pitch)], 
                   [math.cos(roll) * math.sin(pitch), -math.sin(roll), math.cos(roll) * math.cos(pitch)]])

    return T


def sun_centroid_to_rover_heading(u, v, azimuth_astron, camera_matrix, roll=0.0, pitch=0.0, ax=0, ay=0, az=0, static=True):
    camera_matrix_inverse = camera_matrix.I

    S = np.asarray(camera_matrix_inverse @ uv.T) # 3d ray from projection to sun
    S = S / np.linalg.norm(S) # normalize

    print("S:\n", S)


    S_rover = np.matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) @ S # transform to rover frame

    print("S_rover:\n", S_rover)

    if static is True:
        T = rot_matrix_from_roll_pitch(roll, pitch).T
    else:
        T = rot_matrix_from_imu(ax, ay, az)

    print("T:\n", T)

    S_site = T @ S_rover

    print("S_site:\n", S_site)

    # compute ray azimuth and elevation

    azimuth_site = math.atan2(S_site[0], S_site[1])
    elevation_site = math.asin(S_site[2])

    print("azimuth_site:", azimuth_site)
    print("elevation_site:", elevation_site)


    rover_heading = azimuth_astron - azimuth_site if azimuth_astron > azimuth_site else azimuth_site - azimuth_astron

    return rover_heading


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

    print("\ncamera matrix")
    print(camera_matrix)


    while True:
        ret, frame = cap.read()

        if ret is False or frame is None:
            print("could not acquire frame")
            continue

        dt = datetime.datetime.now()

        offset = time.timezone if (time.localtime().tm_isdst == 0) else time.altzone
        offset = offset / 60 / 60 * -1

        when = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, offset)

        # get the sun's apparent location in the sky
        azimuth, elevation = sun_position(when, location, True)

        # output the results
        print("\ndate time:", dt)
        print("location:", location)

        print("\nsun spherical coordinates")
        print("azimuth:", azimuth)
        print("elevation:", elevation)

        if elevation < 0:
            print("elevation < 0, sun not visible")
            continue

        sun_centroid = detect_sun_position(frame, camera_matrix, dist_coefs) # sun centroid

        if sun_centroid is not None:
            u, v = sun_centroid

            # plot detected sun position on the camera frame
            cv2.circle(frame, (u, v), 8, (255, 0, 0), 1)

            # compute rover heading (assume roll = pitch = 0)
            rover_heading = sun_centroid_to_rover_heading(u, v, math.radians(azimuth), camera_matrix)

            print("\nrover heading (deg):", math.degrees(rover_heading))
        else:
            print("\ncouldn't find sun centroid")


        cv2.imshow("camera frame", frame)
        cv2.waitKey(10)
