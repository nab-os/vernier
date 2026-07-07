# VERNIER Python bindings

`pyvernier` exposes the VERNIER C++ library to Python through
[nanobind](https://github.com/wjakob/nanobind) (vendored in `3rdparty/nanobind`).
It covers the common workflow: build a pattern layout, render an image of it,
then detect the pattern and read back its pose.

## Building

Built as part of the normal CMake build, enabled by default
(`-DBUILD_PYTHON_BINDINGS=ON`). Needs the Python 3.8+ development headers
(`python3-dev`); nanobind requires C++17 (only for this target, the rest of the
library stays C++14).

```bash
mkdir -p build && cd build
cmake ..
make pyvernier
```

The module (`pyvernier.cpython-*.so`) lands in `build/python/`, next to
`example.py` and `test_vernier.py`.

## Usage

Images are 2-D `float64` NumPy arrays (mapping onto `Eigen::ArrayXXd`).

```python
import pyvernier as vernier

# Render a periodic pattern at a known pose ...
layout = vernier.PeriodicPatternLayout(15.0, 31, 31)   # period, nRows, nCols
image = layout.renderOrthographicProjection(
    vernier.Pose(6.0, 3.0, 0.2, 2.0), 512, 512)         # x, y, alpha, pixelSize

# ... and recover it.
detector = vernier.PeriodicPatternDetector(15.0)
detector.compute(image)
if detector.patternFound():
    print(detector.get2DPose())
```

## Exposed API

- `Pose` — `x, y, z, alpha, beta, gamma` (2D and 3D constructors).
- `PeriodicPatternDetector`, `MegarenaPatternDetector` — `compute(image)`,
  `patternFound()`, `get2DPose()`, `get3DPose()`, `setBackend(backend)`,
  `getBackend()`, `getPatternPhase()`.
- `PatternPhase` — the phase-retrieval engine (owned by a detector, or usable
  standalone): `compute(image)`, `peaksFound()`, `getPlane1/2()`,
  `getPhase1/2()`, `getUnwrappedPhase1/2()`, `getSpectrum()`,
  `getSpectrumPeak1/2()`, `getPixelPeriod()`, `computePhaseGradients()`,
  backend selection, and the tuning setters/getters (`setSigma`, `setCropFactor`,
  `setMinFrequency`, `setMaxFrequency`, `setMinPeakPower`,
  `setSmoothingKernelSize`, …).
- `PhasePlane` — a fitted phase plane: `getA/B/C()`, `getCoefficients()`,
  `getAngle()`, `getPixelicPeriod()`, `getPhase(y, x)`, `getPosition(...)`,
  `getPositionPixels(...)`.
- `PeriodicPatternLayout` — `renderOrthographicProjection(pose, rows, cols)`.
- `Backend` — compute backend enum (`Backend.CPU`, `Backend.CUDA`).
- `cuda_available()` — `True` if built with CUDA support and a device is present.
- `VernierError` — exception raised for library errors.

Phase maps and spectra are returned as NumPy arrays (`float64` for phase maps,
`complex128` for spectra). Access the detected geometry through the detector's
internal engine:

```python
detector.compute(image)
phase = detector.getPatternPhase()
print(phase.getPixelPeriod())          # detected period, in pixels
print(phase.getPlane1().getAngle())    # pattern orientation, in radians
```

### Units

`PatternPhase` works purely in pixels — `getPixelPeriod()` is the *measured*
period on the sensor, and it is the one independent length the phase engine
knows. The physical scale is supplied to the detector as `physicalPeriod` (e.g.
in µm) and is not re-measured; the two are bridged by the returned pose's
`pixelSize` (physical-unit-per-pixel), so by construction:

```python
pose = detector.get2DPose()
detector.getPhysicalPeriod()                       # physical period, e.g. µm (input)
phase.getPixelPeriod() * pose.pixelSize            # == physicalPeriod
```

Pose translations `pose.x` / `pose.y` are already in the physical unit.

## Compute backend (CPU / CUDA)

The phase-retrieval pipeline runs on the CPU by default. When the library is
built with `-DUSE_CUDA=ON` and a CUDA device is present, detectors can be
switched to the GPU backend:

```python
import pyvernier as vernier

detector = vernier.PeriodicPatternDetector(15.0)
if vernier.cuda_available():
    detector.setBackend(vernier.Backend.CUDA)
detector.compute(image)   # runs on the selected backend
```

`setBackend(Backend.CUDA)` raises `VernierError` if the library was built
without CUDA or no device is available.

## Tests

```bash
cd build/python
python3 -m unittest test_vernier -v
```
