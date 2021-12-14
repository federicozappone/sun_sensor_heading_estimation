# Absolute Heading Estimation Using Sun Sensor

![image](images/camera.jpg)

## Table of contents

- [Quick start](#quick-start)
- [Notes](#notes)
- [Creators](#creators)
- [Copyright and license](#copyright-and-license)

## Quick Start

Install the dependencies:

```
pip install -r requirements.txt
```

Camera calibration:

Print the pattern ```calibration_target/chessboard.png``` on an A4 paper sheet.

Acquire frames of the chessboard using the ```grab_calibration_frames.py``` utility.

Finally, run ```calibrate_camera.py```, if everything goes well you should have a ```calibration.npz``` file inside the ```calibration_data``` folder.

Set your correct location (lat, lon) inside the ```configs/config.yaml``` file.

You can now run the sun sensor script:

```
python sun_sensor.py
```

## Notes

This program has been tested using a Raspberry PI 4 and a camera module with fisheye lens.

I used the standard VideoCapture from OpenCV which supports most USB and RPI cameras.

The program assumes a "static" camera looking upward. 
If you wish to use this in a real world application (e.g. a rover) you need to update the roll and pitch for every frame using an AHRS.

The system could be augmented with a GPS sensor to acquire the current location in real time.

## Creators

**Federico Zappone**

- <https://github.com/federicozappone>

## Copyright and license

Code released under the [MIT License](https://github.com/federicozappone/sun_sensor_heading_estimation/LICENSE.md).
