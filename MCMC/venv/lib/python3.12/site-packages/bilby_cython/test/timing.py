import time

import numpy as np

from bilby.gw.detector import InterferometerList
from bilby_cython import geometry
from bilby_cython.time import greenwich_mean_sidereal_time
from old_code import (
    antenna_response,
    get_polarization_tensor,
    get_polarization_tensor_multiple_modes,
    greenwich_mean_sidereal_time as gmst_old,
    time_delay_geocentric,
    zenith_azimuth_to_theta_phi,
)

ifos = InterferometerList(["H1", "L1"])
detectors = [ifo.vertex for ifo in ifos]

points = np.random.uniform(0, np.pi / 2, (10000, 3))

start = time.time()
for point in points:
    time_delay_geocentric(*detectors, *point)
stop = time.time()
numpy_time = (stop - start) / len(points)
start = time.time()
for point in points:
    geometry.time_delay_geocentric(*detectors, *point)
stop = time.time()
cython_time = (stop - start) / len(points)

print(f"Timing time delay calculation over {len(points)} trials.")
print(f"Cython time: {cython_time:.3e}\nNumpy time: {numpy_time:.3e}")

points = np.random.uniform(0, np.pi / 2, (1000, 4))
modes = ["plus", "cross", "x", "y", "breathing", "longitudinal"]

start = time.time()
for point in points:
    for mode in modes:
        get_polarization_tensor(*point, mode)
stop = time.time()
numpy_time = (stop - start) / len(points) / len(modes)
start = time.time()
for point in points:
    for mode in modes:
        geometry.get_polarization_tensor(*point, mode)
stop = time.time()
cython_time = (stop - start) / len(points) / len(modes)

print(f"Timing polarization tensor calculation over {len(points)} trials.")
print(f"Cython time: {cython_time:.3e}\nNumpy time: {numpy_time:.3e}")

points = np.random.uniform(0, np.pi / 2, (1000, 4))
modes = ["plus", "cross"]

start = time.time()
for point in points:
    get_polarization_tensor_multiple_modes(*point, modes)
stop = time.time()
numpy_time = (stop - start) / len(points) / len(modes)
start = time.time()
for point in points:
    geometry.get_polarization_tensor_multiple_modes(*point, modes)
stop = time.time()
cython_time = (stop - start) / len(points) / len(modes)

print(f"Timing two mode polarization tensor calculation over {len(points)} trials.")
print(f"Cython time: {cython_time:.3e}\nNumpy time: {numpy_time:.3e}")


points = np.random.uniform(0, np.pi / 2, (1000, 4))
modes = ["plus", "cross", "x", "y", "breathing", "longitudinal"]

start = time.time()
for point in points:
    get_polarization_tensor_multiple_modes(*point, modes)
stop = time.time()
numpy_time = (stop - start) / len(points) / len(modes)
start = time.time()
for point in points:
    geometry.get_polarization_tensor_multiple_modes(*point, modes)
stop = time.time()
cython_time = (stop - start) / len(points) / len(modes)

print(f"Timing six mode polarization tensor calculation over {len(points)} trials.")
print(f"Cython time: {cython_time:.3e}\nNumpy time: {numpy_time:.3e}")


def cython_antenna_response(detector_tensor, ra, dec, time, psi, mode):
    polarization = geometry.get_polarization_tensor(ra, dec, time, psi, mode)
    return geometry.three_by_three_matrix_contraction(detector_tensor, polarization)


points = np.random.uniform(0, np.pi / 2, (1000, 4))
modes = ["plus", "cross", "x", "y", "breathing", "longitudinal"]

start = time.time()
for point in points:
    for mode in modes:
        antenna_response(ifos[0].geometry.detector_tensor, *point, mode)
stop = time.time()
numpy_time = (stop - start) / len(points) / len(modes)
start = time.time()
for point in points:
    for mode in modes:
        cython_antenna_response(ifos[0].geometry.detector_tensor, *point, mode)
stop = time.time()
cython_time = (stop - start) / len(points) / len(modes)

print(f"Timing antenna response calculation over {len(points)} trials.")
print(f"Cython time: {cython_time:.3e}\nNumpy time: {numpy_time:.3e}")

points = np.random.uniform(0, np.pi, (1000, 2))
delta_x = ifos[1].geometry.vertex - ifos[0].geometry.vertex
# burn a numpy evaluation for caching
zenith_azimuth_to_theta_phi(*points[0], delta_x)

start = time.time()
for point in points:
    zenith_azimuth_to_theta_phi(*point, delta_x)
stop = time.time()
numpy_time = (stop - start) / len(points)
start = time.time()
for point in points:
    geometry.zenith_azimuth_to_theta_phi(*point, delta_x)
stop = time.time()
cython_time = (stop - start) / len(points)

print(f"Timing frame conversion calculation over {len(points)} trials.")
print(f"Cython time: {cython_time:.3e}\nNumpy time: {numpy_time:.3e}")

times = np.random.uniform(1000000000, 1000100000, 10000)
start = time.time()
for tt in times:
    gmst_old(tt)
stop = time.time()
lal_time = (stop - start) / len(times)
start = time.time()
for tt in times:
    greenwich_mean_sidereal_time(tt)
stop = time.time()
cython_time = (stop - start) / len(times)

print(f"Timing GMST calculation over {len(points)} trials.")
print(f"Cython time: {cython_time:.3e}\nLAL time: {lal_time:.3e}")
