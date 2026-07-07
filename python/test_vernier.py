#
# This file is part of the VERNIER Library.
#
# Copyright (c) 2018-2025 CNRS, ENSMM, UMLP.
#

"""Tests for the VERNIER Python bindings.

Run from the build directory where the module lives:

    cd build/python && python3 -m unittest test_vernier -v
"""

import unittest

import pyvernier as vernier


class TestPose(unittest.TestCase):

    def test_fields(self):
        pose = vernier.Pose(6.0, 3.0, 0.2, 2.0)
        self.assertAlmostEqual(pose.x, 6.0)
        self.assertAlmostEqual(pose.y, 3.0)
        self.assertAlmostEqual(pose.alpha, 0.2)
        self.assertIn("6", repr(pose))


class TestRoundtrip(unittest.TestCase):
    """Render a periodic pattern at a known pose, then check the detector
    recovers it."""

    def test_render_then_detect(self):
        period = 15.0
        layout = vernier.PeriodicPatternLayout(period, 31, 31)
        image = layout.renderOrthographicProjection(vernier.Pose(6.0, 3.0, 0.2, 2.0), 512, 512)

        detector = vernier.PeriodicPatternDetector(period)
        detector.setSigma(1.0)
        detector.setCropFactor(0.4)
        detector.compute(image)

        self.assertTrue(detector.patternFound())
        pose = detector.get2DPose()
        self.assertAlmostEqual(pose.x, 6.0, delta=0.1)
        self.assertAlmostEqual(pose.y, 3.0, delta=0.1)
        self.assertAlmostEqual(pose.alpha, 0.2, delta=0.02)


class TestPatternPhase(unittest.TestCase):
    """Lower-level phase analysis, reached both standalone and through a
    detector's internal PatternPhase."""

    period = 15.0
    pixel_size = 2.0

    def _image(self):
        layout = vernier.PeriodicPatternLayout(self.period, 31, 31)
        return layout.renderOrthographicProjection(
            vernier.Pose(6.0, 3.0, 0.2, self.pixel_size), 512, 512)

    def test_get_pattern_phase_from_detector(self):
        detector = vernier.PeriodicPatternDetector(self.period)
        detector.setSigma(1.0)
        detector.setCropFactor(0.4)
        detector.compute(self._image())

        pp = detector.getPatternPhase()
        self.assertIsInstance(pp, vernier.PatternPhase)
        self.assertTrue(pp.peaksFound())
        self.assertEqual(pp.getNRows(), 512)
        # Period in pixels is the physical period scaled by the pixel size.
        self.assertAlmostEqual(pp.getPixelPeriod(), self.period / self.pixel_size, delta=0.1)

    def test_physical_scale(self):
        # PatternPhase measures the period in pixels; the physical scale (µm) comes
        # from the detector's physicalPeriod, and pose.pixelSize bridges the two:
        # pixelPeriod * pixelSize == physicalPeriod (by construction).
        detector = vernier.PeriodicPatternDetector(self.period)
        detector.setSigma(1.0)
        detector.setCropFactor(0.4)
        detector.compute(self._image())

        self.assertAlmostEqual(detector.getPhysicalPeriod(), self.period)
        pose = detector.get2DPose()
        self.assertAlmostEqual(pose.pixelSize, self.pixel_size, delta=0.05)
        pixel_period = detector.getPatternPhase().getPixelPeriod()
        self.assertAlmostEqual(pixel_period * pose.pixelSize, self.period, delta=0.1)

    def test_planes_and_phase_maps(self):
        pp = vernier.PatternPhase(512, 512)
        pp.setSigma(1.0)
        pp.setCropFactor(0.4)
        pp.compute(self._image())

        plane = pp.getPlane1()
        self.assertIsInstance(plane, vernier.PhasePlane)
        # The plane angle recovers the pattern rotation (alpha = 0.2).
        self.assertAlmostEqual(plane.getAngle(), 0.2, delta=0.02)
        self.assertAlmostEqual(plane.getPixelicPeriod(), pp.getPixelPeriod(), delta=1e-6)

        unwrapped = pp.getUnwrappedPhase1()
        self.assertEqual(unwrapped.shape, (512, 512))
        self.assertEqual(str(unwrapped.dtype), "float64")

        spectrum = pp.getSpectrum()
        self.assertEqual(str(spectrum.dtype), "complex128")

    def test_phase_plane_standalone(self):
        plane = vernier.PhasePlane(0.1, 0.2, 0.3)
        self.assertAlmostEqual(plane.getA(), 0.1)
        plane.setA(0.5)
        self.assertAlmostEqual(plane.a, 0.5)
        self.assertEqual(len(plane.getCoefficients()), 3)


class TestBackend(unittest.TestCase):
    """Compute-backend selection (CPU / CUDA)."""

    def test_default_is_cpu(self):
        detector = vernier.PeriodicPatternDetector(15.0)
        self.assertEqual(detector.getBackend(), vernier.Backend.CPU)

    def test_set_cpu_backend(self):
        detector = vernier.PeriodicPatternDetector(15.0)
        detector.setBackend(vernier.Backend.CPU)
        self.assertEqual(detector.getBackend(), vernier.Backend.CPU)

    def test_cuda_matches_availability(self):
        detector = vernier.PeriodicPatternDetector(15.0)
        if vernier.cuda_available():
            detector.setBackend(vernier.Backend.CUDA)
            self.assertEqual(detector.getBackend(), vernier.Backend.CUDA)
        else:
            # Requesting CUDA without support must raise and leave the backend
            # unchanged.
            with self.assertRaises(vernier.VernierError):
                detector.setBackend(vernier.Backend.CUDA)
            self.assertEqual(detector.getBackend(), vernier.Backend.CPU)


if __name__ == "__main__":
    unittest.main()
