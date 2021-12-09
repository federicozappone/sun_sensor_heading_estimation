import numpy as np
import datetime
import time
import math
import cv2

from scipy.spatial.transform import Rotation


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

    # return azimuth and elevation in degrees
    return (round(azimuth, 2), round(elevation, 2))


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix


def detect_sun_position(image):
    # to be defined
    return 300, 200


if __name__ == "__main__":
    location = (41.4622884, 12.8663671) # latina, italy

    while True:
        dt = datetime.datetime.now()

        offset = time.timezone if (time.localtime().tm_isdst == 0) else time.altzone
        offset = offset / 60 / 60 * -1

        when = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, offset)
        # Get the Sun's apparent location in the sky
        azimuth, elevation = sun_position(when, location, True)
        # Output the results
        print("\ndate time:", when)
        print("location:", location)

        print("\nsun spherical coordinates")
        print("azimuth:", azimuth)
        print("elevation:", elevation)

        sun_distance = 150 * 1e9 # meters

        x, y, z = spherical_to_cartesian(sun_distance, math.radians(elevation), math.radians(azimuth))

        z = abs(z)

        print("\nsun cartesian coordinates")
        print("x:", x)
        print("y:", y)
        print("z:", z)

        w = 640
        h = 480

        image = np.zeros((h, w, 3), dtype=np.uint8)

        # this needs to be correctly estimated 
        # using camera calibration
        px = w / 2
        py = h / 2
        fx = 50
        fy = 50

        I = np.eye(3)
        t = np.zeros(3)
        R = Rotation.from_euler("y", 90, degrees=True)

        translation_2d = I
        camera_matrix = np.matrix([[fx, 0, px], [0, fy, py], [0, 0, 1]])
        shear_2d = I

        translation_3d = np.c_[I, t]
        rotation_3d = np.c_[np.r_[R.as_matrix(), [np.zeros(3)]], [0, 0, 0, 1]]

        print("\ncamera 3d translation")
        print(translation_3d)
        print("camera 3d rotation")
        print(rotation_3d)

        P = np.array([x, y, z, 1]).T

        # assuming undistorted camera
        p = translation_2d @ camera_matrix @ shear_2d @ translation_3d @ rotation_3d @ P
        p /= p[0, 2]

        print("\nprojected sun position in image plane:", p)

        u_est = int(p[0, 0])
        v_est = int(p[0, 1])

        if u_est > w or v_est > h:
            print("sun projection outside of image place")
            continue

        u_real, v_real = detect_sun_position(image)

        # plot estimated sun position
        cv2.circle(image, (u_est, v_est), 4, (255, 255, 255), -1)
        cv2.line(image, (w // 2, h // 2), (u_est, v_est), (255, 255, 255), 1)

        # plot detected sun position
        cv2.circle(image, (u_real, v_real), 4, (0, 255, 0), -1)
        cv2.line(image, (w // 2, h // 2), (u_real, v_real), (0, 255, 0), 1)

        # compute rotation matrix from one vector to another
        rot = rotation_matrix_from_vectors([u_est, v_est, 0], [u_real, v_real, 0])
        
        rot = Rotation.from_matrix(rot)

        print("\ncamera estimated rotation matrix")
        print(rot.as_matrix())

        print("camera estimated rotation vector")
        print(rot.as_rotvec())

        camera_heading = azimuth + math.degrees(rot.as_rotvec()[2])

        print("\ncamera heading (deg):", camera_heading)

        cv2.imshow("image", image)
        cv2.waitKey(500)
