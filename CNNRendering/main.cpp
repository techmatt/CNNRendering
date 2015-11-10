
#include "main.h"

int main(int argc, const char** argv)
{
    /*DepthImage d(2, 2);
    
    d(0, 0) = 1.0f;
    d(1, 0) = 6.0f;
    d(0, 1) = 6.0f;
    d(1, 1) = 100.0f;

    d.save("C:\\code\\test.dat", true);
    d.load("C:\\code\\test.dat", true);
    d.save("C:\\code\\test2.dat", true);
    d.load("C:\\code\\test2.dat", true);*/

    ReconstructionParams reconParams;

    CNN cnn;
    cnn.initStandard();
    
    Bitmap testImage, reconstructedImage;
    LayerData testOutput;

    const string dataDir = "../data/";
    const string imageDir = "../testImages/";
    const string outputDir = "../testResults/";

    testImage = ml::LodePNG::load(imageDir + "imageA.png");
    cnn.filter(testImage, testOutput);

    cnn.layer.invert(reconParams, testOutput, cnn.transform.meanValues, reconstructedImage);

    ml::LodePNG::save(reconstructedImage, outputDir + reconParams.toString() + ".png");

    for (UINT filter = 0; filter < testOutput.images.size(); filter++)
    {
        const Bitmap bmp = testOutput.images[filter].makeVisualization(reconParams);
        ml::LodePNG::save(bmp, outputDir + util::zeroPad(filter, 2) + ".png");
    }

    return 0;
}
