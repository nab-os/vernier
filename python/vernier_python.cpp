/*
 * This file is part of the VERNIER Library.
 *
 * Copyright (c) 2018-2025 CNRS, ENSMM, UMLP.
 */

/** \file vernier_python.cpp
 *
 *  Python bindings for the VERNIER library, built with nanobind.
 *
 *  This exposes the classes needed for the common workflow: build a pattern
 *  layout, render an image of it, then detect the pattern and read back its
 *  pose. Images are exchanged as 2-D float64 NumPy arrays, which map directly
 *  onto the Eigen::ArrayXXd used by the library.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/eigen/dense.h>

#include "Vernier.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace vernier;

NB_MODULE(pyvernier, m) {

    m.doc() = "Python bindings for the VERNIER pose-measurement library.";

    // Library errors surface as `pyvernier.VernierError`.
    nb::exception<Exception>(m, "VernierError");

    // ─── Compute backend ───────────────────────────────────────────────────────

    // Selects where the phase-retrieval pipeline runs. CUDA is only usable when
    // the library was built with -DUSE_CUDA=ON and a device is present at runtime.
    nb::enum_<Backend>(m, "Backend", "Compute backend for phase retrieval.")
        .value("CPU", Backend::CPU)
        .value("CUDA", Backend::CUDA);

    m.def("cuda_available", &PatternPhase::cudaAvailable,
        "True if the library was built with CUDA support and a device is present.");

    // ─── Pose ─────────────────────────────────────────────────────────────────

    nb::class_<Pose>(m, "Pose", "Pose of a pattern: translations and rotations.")
        .def(nb::init<>())
        .def(nb::init<double, double, double, double>(),
            "x"_a, "y"_a, "alpha"_a, "pixelSize"_a = 1.0,
            "2D pose from x, y and the rotation alpha about the Z axis.")
        .def(nb::init<double, double, double, double, double, double, double>(),
            "x"_a, "y"_a, "z"_a, "alpha"_a, "beta"_a, "gamma"_a, "pixelSize"_a = 1.0,
            "3D pose from x, y, z and the intrinsic rotations alpha, beta, gamma.")
        .def_rw("x", &Pose::x)
        .def_rw("y", &Pose::y)
        .def_rw("z", &Pose::z)
        .def_rw("alpha", &Pose::alpha)
        .def_rw("beta", &Pose::beta)
        .def_rw("gamma", &Pose::gamma)
        // Pixel-to-physical scale (e.g. µm/pixel); x, y are in the same physical
        // unit as the pattern's period.
        .def_rw("pixelSize", &Pose::pixelSize)
        .def("__repr__", &Pose::toString);

    // ─── Detectors ─────────────────────────────────────────────────────────────

    nb::class_<PatternDetector>(m, "PatternDetector")
        .def("compute", nb::overload_cast<const Eigen::ArrayXXd&>(&PatternDetector::compute),
            "image"_a, "Detects the pattern in a 2-D float64 image array.")
        .def("patternFound", &PatternDetector::patternFound, "id"_a = -1)
        .def("get2DPose", &PatternDetector::get2DPose, "id"_a = -1)
        .def("get3DPose", &PatternDetector::get3DPose, "id"_a = -1)
        .def("__repr__", &PatternDetector::toString);

    nb::class_<PeriodicPatternDetector, PatternDetector>(m, "PeriodicPatternDetector")
        .def(nb::init<double>(), "physicalPeriod"_a = 1.0)
        .def("setSigma", &PeriodicPatternDetector::setSigma, "sigma"_a)
        .def("setCropFactor", &PeriodicPatternDetector::setCropFactor, "cropFactor"_a)
        // The physical period (µm) supplied at construction — the scale that turns
        // the measured pixel period into a physical length.
        .def("getPhysicalPeriod", &PeriodicPatternDetector::getPhysicalPeriod,
            "Returns the pattern's physical period (in the unit it was given, e.g. µm).")
        .def("setPhysicalPeriod", &PeriodicPatternDetector::setPhysicalPeriod,
            "physicalPeriod"_a)
        // The backend lives on the detector's internal PatternPhase; forward to it
        // so callers can pick CPU/CUDA without reaching into the phase engine.
        .def("setBackend",
            [](PeriodicPatternDetector& self, Backend backend) {
                self.getPatternPhase()->setBackend(backend);
            },
            "backend"_a,
            "Selects the compute backend. Raises VernierError if CUDA is requested "
            "but unavailable.")
        .def("getBackend",
            [](PeriodicPatternDetector& self) {
                return self.getPatternPhase()->getBackend();
            },
            "Returns the currently selected compute backend.")
        // Exposes the detector's internal phase engine for lower-level access
        // (planes, phase maps, detected period, tuning). The returned object is
        // owned by the detector; its lifetime is tied to it.
        .def("getPatternPhase", &PeriodicPatternDetector::getPatternPhase,
            nb::rv_policy::reference_internal,
            "Returns the detector's internal PatternPhase (owned by the detector).");

    nb::class_<MegarenaPatternDetector, PeriodicPatternDetector>(m, "MegarenaPatternDetector")
        .def(nb::init<double, int>(), "physicalPeriod"_a, "codeSize"_a);

    // ─── Phase analysis ─────────────────────────────────────────────────────────

    // A fitted phase plane `a*x + b*y + c = z`, returned by PatternPhase.
    nb::class_<PhasePlane>(m, "PhasePlane",
        "A fitted phase plane (a*x + b*y + c = z) of a pattern direction.")
        .def(nb::init<>())
        .def(nb::init<double, double, double>(), "a"_a, "b"_a, "c"_a)
        .def_rw("a", &PhasePlane::a)
        .def_rw("b", &PhasePlane::b)
        .def_rw("c", &PhasePlane::c)
        .def("getCoefficients", &PhasePlane::getCoefficients,
            "Returns the plane coefficients [a, b, c].")
        .def("getA", &PhasePlane::getA)
        .def("getB", &PhasePlane::getB)
        .def("getC", &PhasePlane::getC)
        .def("setA", &PhasePlane::setA, "a"_a)
        .def("setB", &PhasePlane::setB, "b"_a)
        .def("setC", &PhasePlane::setC, "c"_a)
        .def("getPhase", &PhasePlane::getPhase, "y"_a = 0.0, "x"_a = 0.0,
            "Phase at position (y, x) (default: image center).")
        .def("getAngle", &PhasePlane::getAngle, "Angle of the plane in radians.")
        .def("getPosition", &PhasePlane::getPosition,
            "physicalPeriod"_a, "y"_a = 0.0, "x"_a = 0.0, "periodShift"_a = 0,
            "Position in the same unit as the period length.")
        .def("getPositionPixels", &PhasePlane::getPositionPixels,
            "y"_a = 0.0, "x"_a = 0.0, "periodShift"_a = 0,
            "Position in pixels.")
        .def("getPixelicPeriod", &PhasePlane::getPixelicPeriod,
            "Pixelic period of the pattern along this direction.")
        .def("flip", &PhasePlane::flip)
        .def("turnClockwise90", &PhasePlane::turnClockwise90)
        .def("turnAntiClockwise90", &PhasePlane::turnAntiClockwise90)
        .def("turn180", &PhasePlane::turn180)
        .def("__repr__", &PhasePlane::toString);

    // The phase-retrieval engine. Detectors own one (see getPatternPhase); it can
    // also be driven directly for lower-level phase analysis.
    nb::class_<PatternPhase>(m, "PatternPhase",
        "Computes the phase planes of a pattern image.")
        .def(nb::init<>())
        .def(nb::init<int, int>(), "nRows"_a, "nCols"_a)
        .def("resize", &PatternPhase::resize, "nRows"_a, "nCols"_a)
        .def("compute", nb::overload_cast<const Eigen::ArrayXXd&>(&PatternPhase::compute),
            "image"_a, "Computes the phase planes of a 2-D float64 image array.")
        .def("peaksFound", &PatternPhase::peaksFound,
            "True if two peaks with sufficient power were found.")
        .def("computePhaseGradients",
            [](PatternPhase& self) {
                int betaSign = 0, gammaSign = 0;
                self.computePhaseGradients(betaSign, gammaSign);
                return std::make_pair(betaSign, gammaSign);
            },
            "Returns the (betaSign, gammaSign) of the out-of-plane angles.")
        // Backend selection (see also the module-level `cuda_available`).
        .def("setBackend", &PatternPhase::setBackend, "backend"_a,
            "Selects the compute backend. Raises VernierError if CUDA is "
            "requested but unavailable.")
        .def("getBackend", &PatternPhase::getBackend,
            "Returns the currently selected compute backend.")
        // Results.
        .def("getPlane1", &PatternPhase::getPlane1, "First phase plane.")
        .def("getPlane2", &PatternPhase::getPlane2, "Second phase plane.")
        .def("getPhase1", &PatternPhase::getPhase1, "First wrapped phase map.")
        .def("getPhase2", &PatternPhase::getPhase2, "Second wrapped phase map.")
        .def("getUnwrappedPhase1", &PatternPhase::getUnwrappedPhase1,
            "First unwrapped phase map.")
        .def("getUnwrappedPhase2", &PatternPhase::getUnwrappedPhase2,
            "Second unwrapped phase map.")
        .def("getSpectrum", &PatternPhase::getSpectrum, "Shifted spectrum.")
        .def("getSpectrumPeak1", &PatternPhase::getSpectrumPeak1,
            "Spectrum filtered around peak 1.")
        .def("getSpectrumPeak2", &PatternPhase::getSpectrumPeak2,
            "Spectrum filtered around peak 2.")
        .def("getPixelPeriod", &PatternPhase::getPixelPeriod,
            "Detected period length in pixels.")
        // Tuning.
        .def("setSigma", &PatternPhase::setSigma, "sigma"_a)
        .def("getSigma", &PatternPhase::getSigma)
        .def("setCropFactor", &PatternPhase::setCropFactor, "cropFactor"_a)
        .def("setMinFrequency", &PatternPhase::setMinFrequency, "minFrequency"_a)
        .def("getMinFrequency", &PatternPhase::getMinFrequency)
        .def("setMaxFrequency", &PatternPhase::setMaxFrequency, "maxFrequency"_a)
        .def("getMaxFrequency", &PatternPhase::getMaxFrequency)
        .def("setMinPeakPower", &PatternPhase::setMinPeakPower, "minPeakPower"_a)
        .def("getMinPeakPower", &PatternPhase::getMinPeakPower)
        .def("setSmoothingKernelSize", &PatternPhase::setSmoothingKernelSize,
            "smoothingKernelSize"_a)
        .def("getSmoothingKernelSize", &PatternPhase::getSmoothingKernelSize)
        .def("getNRows", &PatternPhase::getNRows)
        .def("getNCols", &PatternPhase::getNCols)
        .def("rotate90", &PatternPhase::rotate90)
        .def("rotate180", &PatternPhase::rotate180)
        .def("rotate270", &PatternPhase::rotate270);

    // ─── Layouts (for rendering test images) ───────────────────────────────────

    nb::class_<PatternLayout>(m, "PatternLayout")
        // The C++ API renders in place; here we allocate the array and return it.
        .def("renderOrthographicProjection",
            [](PatternLayout& self, Pose pose, int rows, int cols) {
                Eigen::ArrayXXd out(rows, cols);
                self.renderOrthographicProjection(pose, out);
                return out;
            },
            "pose"_a, "rows"_a, "cols"_a,
            "Renders a `rows x cols` image of the layout at the given pose.");

    nb::class_<PeriodicPatternLayout, PatternLayout>(m, "PeriodicPatternLayout")
        .def(nb::init<double, int, int>(), "period"_a, "nRows"_a, "nCols"_a);
}
