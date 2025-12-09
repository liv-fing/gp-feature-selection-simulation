import itertools
import unittest

import numpy as np
from bilby.gw.detector import get_empty_interferometer, InterferometerList
from bilby_cython import geometry

from .old_code import (
    antenna_response,
    calculate_arm,
    get_polarization_tensor,
    get_polarization_tensor_multiple_modes,
    time_delay_geocentric,
    zenith_azimuth_to_theta_phi,
)


class TestGeometry(unittest.TestCase):

    def setUp(self) -> None:
        self.ifos = ["H1", "L1", "V1", "K1"]
        self.modes = ["plus", "cross", "x", "y", "breathing", "longitudinal"]

    def tearDown(self) -> None:
        pass

    def test_time_delay(self):
        max_diff = 0
        for ifo_pair in itertools.product(self.ifos, repeat=2):
            if ifo_pair[0] == ifo_pair[1]:
                continue
            ifos = InterferometerList(ifo_pair)
            detectors = [ifo.vertex for ifo in ifos]
            for point in np.random.uniform(0, np.pi / 2, (1000, 3)):
                numpy_delay = time_delay_geocentric(*detectors, *point)
                cython_delay = geometry.time_delay_geocentric(*detectors, *point)
                max_diff = max(max_diff, abs(numpy_delay - cython_delay))
        self.assertLess(max_diff, 1e-6)

    def test_time_delay_from_geocenter_matches_time_delay_geocentric(self):
        geocenter = np.zeros(3)
        for ifo in self.ifos:
            detector = get_empty_interferometer(ifo).vertex
            for point in np.random.uniform(0, np.pi / 2, (1000, 3)):
                geocentric = geometry.time_delay_geocentric(detector, geocenter, *point)
                from_geocenter = geometry.time_delay_from_geocenter(detector, *point)
                self.assertEqual(from_geocenter, geocentric)

    def test_get_polarization_tensor(self):
        max_diff = 0
        for ra, dec, time, psi in np.random.uniform(0, np.pi / 2, (100, 4)):
            for mode in self.modes:
                args = (ra, dec, time, psi, mode)
                numpy_tensor = get_polarization_tensor(*args)
                cython_tensor = geometry.get_polarization_tensor(*args)
                max_diff = abs(np.max(numpy_tensor - cython_tensor))
        self.assertLess(max_diff, 1e-8)

    def test_get_polarization_tensor_multiple_modes(self):
        max_diff = 0
        for ra, dec, time, psi in np.random.uniform(0, np.pi / 2, (100, 4)):
            args = (ra, dec, time, psi, self.modes)
            numpy_tensors = get_polarization_tensor_multiple_modes(*args)
            cython_tensors = geometry.get_polarization_tensor_multiple_modes(*args)
            max_diff = abs(np.max(np.array(numpy_tensors) - np.array(cython_tensors)))
        self.assertLess(max_diff, 1e-8)

    def test_polarization_tensor_bad_mode_raises_error(self):
        with self.assertRaises(ValueError):
            _ = geometry.get_polarization_tensor(0.0, 0.0, 0.0, 0.0, "bad_mode")

    def test_antenna_reponse(self):
        def _cython_antenna_response(detector_tensor, *args):
            polarization = geometry.get_polarization_tensor(*args)
            return geometry.three_by_three_matrix_contraction(detector_tensor, polarization)

        max_diff = 0
        for ifo in self.ifos:
            detector = get_empty_interferometer(ifo).geometry.detector_tensor
            for ra, dec, time, psi in np.random.uniform(0, np.pi / 2, (100, 4)):
                for mode in self.modes:
                    args = (ra, dec, time, psi, mode)
                    numpy_tensor = antenna_response(detector, *args)
                    cython_tensor = _cython_antenna_response(detector, *args)
                    max_diff = abs(np.max(numpy_tensor - cython_tensor))
        self.assertLess(max_diff, 1e-7)

    def test_frame_conversion(self):
        max_diff = 0
        for ifo_pair in itertools.product(self.ifos, repeat=2):
            if ifo_pair[0] == ifo_pair[1]:
                continue
            ifos = InterferometerList(ifo_pair)
            delta_x = ifos[0].vertex - ifos[1].vertex
            # there is a bug in the python implementation for separations
            # lying in specific quadrants for arctan2 due to using arctan
            # the direct tests of the rotation matrix are more rigorous
            sign_incorrect = abs(np.arctan2(delta_x[1], delta_x[0])) < np.pi / 2
            delta_x *= (-1)**sign_incorrect
            for point in np.random.uniform(0, np.pi / 2, (100, 2)):
                numpy_result = zenith_azimuth_to_theta_phi(*point, delta_x)
                cython_result = geometry.zenith_azimuth_to_theta_phi(*point, delta_x)
                max_diff = max(max_diff, np.max(abs(np.array(numpy_result) - np.array(cython_result))))
        self.assertLess(max_diff, 1e-6)

    def test_calculate_arm(self):
        max_diff = 0
        for point in np.random.uniform(0, 2 * np.pi, (1000, 4)):
            print(geometry.calculate_arm(*point), calculate_arm(*point))
            max_diff = max(max_diff, np.linalg.norm(
                geometry.calculate_arm(*point) - calculate_arm(*point)
            ))
        self.assertLess(max_diff, 1e-10)

    def test_detector_tensor(self):
        for xx, yy in np.random.uniform(0, 1, (1000, 2, 3)):
            numpy_tensor = 0.5 * (np.einsum('i,j->ij', xx, xx) - np.einsum('i,j->ij', yy, yy))
            cython_tensor = geometry.detector_tensor(xx, yy)
            self.assertAlmostEqual(np.min(numpy_tensor - cython_tensor), 0, 6)

    def test_rotation_matrix_transpose_is_inverse(self):
        for delta_x in np.random.uniform(0, 1, (100, 3)):
            rotation = geometry.rotation_matrix_from_delta(delta_x)
            self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3)))

    def test_rotation_matrix_maps_delta_x_to_z_axis(self):
        for delta_x in np.random.uniform(0, 1, (100, 3)):
            rotation = geometry.rotation_matrix_from_delta(delta_x)
            delta_x /= np.linalg.norm(delta_x)
            self.assertTrue(np.allclose(rotation.T @ delta_x, np.array([0, 0, 1])))
