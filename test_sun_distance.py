import datetime
import math

day_of_the_year = datetime.datetime.now().timetuple().tm_yday
au = 149597870700 # meters
sun_distance = 1.0 - 0.01672 * math.cos(((2 * math.pi) / 365.256363) * (day_of_the_year - 4))

print(sun_distance * au)