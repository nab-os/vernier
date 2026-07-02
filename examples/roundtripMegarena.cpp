#include "Vernier.hpp"

#include <iomanip>
#include <cmath>

using namespace vernier;
using namespace std;
using namespace Eigen;

// Roundtrip precision test: render a megarena pattern at a known pose,
// detect it, and see how close we get back.

// Wrap a length residual into [-period/2, +period/2].
static double wrapHalf(double d, double period) {
    d = fmod(d, period);
    if (d < 0.0) d += period;
    if (d > period / 2.0) d -= period;
    return d;
}

int main(int argc, char** argv) {

    // Pose is camera->pattern, so the coded region at positive coords needs
    // negative translations to come into view (in µm, like the period).
    double trueX = -6000.7;
    double trueY = -8000.3;
    double alpha = 0.15;    // rad
    double pixelSize = 2.0; // µm/px  ->  14 µm period = 7 px
    int    width = 512;
    int    height = 512;

    // Optional overrides: roundtripMegarena [x] [y] [alpha] [pixelSize]
    if (argc > 1) trueX = atof(argv[1]);
    if (argc > 2) trueY = atof(argv[2]);
    if (argc > 3) alpha = atof(argv[3]);
    if (argc > 4) pixelSize = atof(argv[4]);

    unique_ptr<PatternLayout> layout(Layout::loadFromJSON("megarenaPattern.json"));
    double period = layout->getDouble("period"); // µm
    int    codeSize = layout->getInt("codeDepth");
    cout << "period=" << period << " µm  codeSize=" << codeSize
         << "  pixelSize=" << pixelSize << " µm/px  ("
         << period / pixelSize << " px/period)" << endl;

    // Render at the known pose.
    Pose truePose = Pose(trueX, trueY, alpha, pixelSize);
    ArrayXXd image(height, width);
    layout->renderOrthographicProjection(truePose, image);

    // Detect.
    MegarenaPatternDetector detector(period, codeSize);
    detector.compute(image);
    if (!detector.patternFound()) {
        cout << "Pattern not found — try a larger image or different pose." << endl;
        return 1;
    }
    Pose rec = detector.get2DPose();

    // Compare. Absolute error covers the whole decode; fine error drops the
    // period ambiguity to show the sub-period precision.
    const double NM = 1000.0; // µm -> nm
    double absX = fabs(rec.x - trueX);
    double absY = fabs(rec.y - trueY);
    double fineX = fabs(wrapHalf(rec.x - trueX, period));
    double fineY = fabs(wrapHalf(rec.y - trueY, period));

    double dAlpha = rec.alpha - alpha;
    while (dAlpha >  M_PI) dAlpha -= 2.0 * M_PI;
    while (dAlpha < -M_PI) dAlpha += 2.0 * M_PI;

    cout << fixed;
    cout << "\ntrue:      x=" << setprecision(4) << trueX << "µm  y=" << trueY
         << "µm  theta=" << setprecision(6) << alpha << " rad" << endl;
    cout << "recovered: x=" << setprecision(4) << rec.x << "µm  y=" << rec.y
         << "µm  theta=" << setprecision(6) << rec.alpha << " rad" << endl;
    cout << "error abs:  Dx=" << setprecision(1) << absX * NM << "nm  Dy=" << absY * NM
         << "nm  Dtheta=" << scientific << setprecision(2) << fabs(dAlpha)
         << " rad" << fixed << endl;
    cout << "error fine: Dx=" << setprecision(1) << fineX * NM << "nm  Dy=" << fineY * NM
         << "nm" << endl;

    return 0;
}
