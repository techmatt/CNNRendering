
#include "Main.h"

#include "mLibEigen.h"

ReconstructionParams::ReconstructionParams()
{
    quartile = 0.25f;
    smoothnessWeight = 2.0f;
    regularizationWeight = 0.5f;
}

Bitmap LayerImage::makeVisualization() const
{
    Bitmap result((int)values.getDimX(), (int)values.getDimY());
    result.fill([&](size_t x, size_t y) {
        const float v = values(x, y);

        BYTE c = ml::util::boundToByte((v + 128.0f));
        if (v == 0.0f)
            c = 0;

        return ml::vec4uc(c, c, c, 255);
    });
    return result;
}

Bitmap LayerImage::makeVisualization(const ReconstructionParams &params) const
{
    float cutoff = comptueCutoff(params);

    Bitmap result((int)values.getDimX(), (int)values.getDimY());
    result.fill([&](size_t x, size_t y) {
        const float v = values(x, y);

        BYTE c = ml::util::boundToByte((v + 128.0f));
        if (v <= cutoff)
            c = 0;

        return ml::vec4uc(c, c, c, 255);
    });
    return result;
}

float LayerImage::comptueCutoff(const ReconstructionParams &params) const
{
    vector<float> v;
    for (auto &e : values)
        if (e.value > 0.0f)
            v.push_back(e.value);

    if (v.size() <= 10)
        return 0.0f;

    std::sort(v.begin(), v.end());
    return v[size_t(params.quartile * v.size())];
}

float Filter::filter(const LayerData &layer, int xStart, int yStart) const
{
    float sum = -bias;
    for (int channel = 0; channel < 3; channel++)
    {
        for (int x = 0; x < 11; x++)
        {
            for (int y = 0; y < 11; y++)
            {
                sum += values(channel, x, y) * layer.images[channel].values(x + xStart, y + yStart);
            }
        }
    }
    return std::max(sum, 0.0f);
}

vector< pair<UINT, float> > Filter::makeFilterRow(int xStart, int yStart, int yPitch, int channelPitch) const
{
    vector< pair<UINT, float> > result;
    for (int channel = 0; channel < 3; channel++)
    {
        for (int x = 0; x < 11; x++)
        {
            for (int y = 0; y < 11; y++)
            {
                float value = values(channel, x, y);
                UINT pixelIndex = channel * channelPitch + (y + yStart) * yPitch + x + xStart;
                result.push_back(make_pair(pixelIndex, value));
            }
        }
    }
    return result;
}

Bitmap Filter::makeVisualization() const
{
    auto rescale = [](float value) -> BYTE {
        float v = (value + 0.5f) * 255.0f;
        if (v >= 255.0f) return 255;
        if (v <= 0.0f) return 0;
        return BYTE(v);
    };

    Bitmap filterViz(11, 11);
    filterViz.fill([&](size_t x, size_t y) {
        return RGBColor( rescale(values(0, x, y)), rescale(values(1, x, y)), rescale(values(2, x, y)));
    });

    const UINT borderSize = 1;
    Bitmap result(11 + borderSize * 2, 11 + borderSize * 2);
    result.copyIntoImage(filterViz, borderSize, borderSize);

    return result;
}

Bitmap FilterBank::makeVisualization() const
{
    const UINT filterXCount = 12;
    const UINT filterYCount = 8;
    
    const Bitmap testViz = filters[0].makeVisualization();
    const UINT filterVizWidth = testViz.getWidth();
    const UINT filterVizHeight = testViz.getHeight();

    Bitmap result(filterXCount * filterVizWidth, filterYCount * filterVizHeight);

    UINT filterIndex = 0;
    for (UINT filterX = 0; filterX < filterXCount; filterX++)
    {
        for (UINT filterY = 0; filterY < filterYCount; filterY++)
        {
            const Bitmap filterViz = filters[filterIndex++].makeVisualization();
            result.copyIntoImage(filterViz, filterX * filterVizWidth, filterY * filterVizHeight);
        }
    }

    return result;
}

void FilterBank::loadFromBlob(const string &filterBlobFile, const string &biasBlobFile)
{
    const UINT filterCount = 96;
    const UINT filterSize = 11;

    stride = 4;
    filters.resize(filterCount);
    for (Filter &f : filters)
        f.values.allocate(3, filterSize, filterSize);

    for (const string &s : ml::util::getFileLines(filterBlobFile))
    {
        const auto words = ml::util::split(s, ' ');
        const int num = ml::convert::toInt(words[0]);
        const int channel = ml::convert::toInt(words[1]);
        const int width = ml::convert::toInt(words[2]);
        const int height = ml::convert::toInt(words[3]);
        const float value = ml::convert::toFloat(words[4]);

        filters[num].values(channel, width, height) = value;
    }

    for (const string &s : ml::util::getFileLines(filterBlobFile))
    {
        const auto words = ml::util::split(s, ' ');
        const int index = ml::convert::toInt(words[3]);
        const float bias = ml::convert::toFloat(words[4]);

        filters[index].bias = bias;
    }
}

void DataTransform::loadFromBlob(const string &blobFile)
{
    meanValues.allocate(3, 256, 256);
    for (const string &s : ml::util::getFileLines(blobFile))
    {
        auto words = ml::util::split(s, ' ');
        int num = ml::convert::toInt(words[0]);
        int channel = ml::convert::toInt(words[1]);
        int width = ml::convert::toInt(words[2]);
        int height = ml::convert::toInt(words[3]);
        float value = ml::convert::toFloat(words[4]);

        meanValues(channel, width, height) = value;
    }
}

void DataTransform::transform(const Bitmap &input, LayerData &output) const
{
    //
    // CNNs work on the inner-most 227 pixels, but the input should still be 256x256
    // the paper claims 224 pixels, but if you do the math on the 1st filter layer, I think it really should be 227...
    //
    const UINT imageSize = 227;
    const UINT offset = 15;

    output.images.resize(3);
    for (int channel = 0; channel < 3; channel++)
    {
        output.images[channel].values = Grid2<float>(imageSize, imageSize, [&](size_t x, size_t y) {
            return float(input(x + offset, y + offset)[channel]) - meanValues(channel, x + offset, y + offset);
        });
    }
}

void CNNLayer::filter(const LayerData &input, LayerData &output) const
{
    const UINT filterCount = (UINT)filters.filters.size();
    const UINT imageSize = 55;

    output.images.resize(filterCount);
    for (UINT filter = 0; filter < filterCount; filter++)
    {
        output.images[filter].values.allocate(imageSize, imageSize);
        for (int filterX = 0; filterX < 55; filterX++)
        {
            for (int filterY = 0; filterY < 55; filterY++)
            {
                output.images[filter].values(filterX, filterY) = filters.filters[filter].filter(input, filterX * filters.stride, filterY * filters.stride);
            }
        }
    }
}

void CNNLayer::invert(const ReconstructionParams &params, const LayerData &nextLayer, const Grid3<float> &imageNetMean, Bitmap &reconstructedImage) const
{
    const size_t filterCount = filters.filters.size();
    const size_t imageDimension = 227;
    const size_t yPitch = imageDimension;
    const size_t channelPitch = imageDimension * imageDimension;
    const size_t pixelCount = imageDimension * imageDimension;

    auto mapPixelToArrray = [&](int channel, int x, int y) -> UINT {
        return UINT(channel * channelPitch + y * yPitch + x);
    };

    //
    // f = (constraintCount x pixelCount*3) filter matrix
    // p = (pixelCount*3 x 1) pixel matrix
    // r = (constraintCount x 1) filter result matrix
    //
    // f * p = r
    // fT f p = fT r
    // p = solve(fT f, fT r)
    //
    
    size_t filterConstraintCount = 0;
    for (size_t filterIndex = 0; filterIndex < filterCount; filterIndex++)
    {
        const LayerImage &nextImage = nextLayer.images[filterIndex];
        const float cutoff = nextImage.comptueCutoff(params);

        for (int filterX = 0; filterX < 55; filterX++)
            for (int filterY = 0; filterY < 55; filterY++)
                if (nextImage.values(filterX, filterY) > cutoff)
                    filterConstraintCount++;
    }

    const size_t smoothnessConstraintCount = 2 * 3 * (imageDimension - 1) * (imageDimension - 1);

    const size_t regularizationConstraintCount = 3 * pixelCount;

    const size_t constraintCount = filterConstraintCount + smoothnessConstraintCount + regularizationConstraintCount;
    cout << "Constraint count: " << constraintCount << endl;

    SparseMatrixf f(constraintCount, pixelCount * 3);
    ml::MathVector<float> r(constraintCount, 1);

    //
    // Filter constraints
    //
    size_t constraintIndex = 0;
    size_t nnz = 0;
    for (size_t filterIndex = 0; filterIndex < filterCount; filterIndex++)
    {
        const Filter &filter = filters.filters[filterIndex];
        
        const LayerImage &nextImage = nextLayer.images[filterIndex];
        const float cutoff = nextImage.comptueCutoff(params);

        for (int filterX = 0; filterX < 55; filterX++)
            for (int filterY = 0; filterY < 55; filterY++)
                if (nextImage.values(filterX, filterY) > cutoff)
                {
                    for (const auto &p : filter.makeFilterRow(filterX * filters.stride, filterY * filters.stride, yPitch, channelPitch))
                    {
                        f.insert((UINT)constraintIndex, p.first, p.second);
                        nnz++;
                    }
                    r[constraintIndex++] = nextLayer.images[filterIndex].values(filterX, filterY) + filter.bias;
                }
    }


    //
    // Smoothness constraints
    //
    for (int channel = 0; channel < 3; channel++)
    {
        for (int x = 0; x < imageDimension - 1; x++)
        {
            for (int y = 0; y < imageDimension - 1; y++)
            {
                //
                // horizontal smoothness
                //
                f.insert((UINT)constraintIndex, mapPixelToArrray(channel, x, y), params.smoothnessWeight);
                f.insert((UINT)constraintIndex, mapPixelToArrray(channel, x + 1, y), -params.smoothnessWeight);
                r[constraintIndex++] = 0.0f;

                // vertical smoothness
                //
                f.insert((UINT)constraintIndex, mapPixelToArrray(channel, x, y), params.smoothnessWeight);
                f.insert((UINT)constraintIndex, mapPixelToArrray(channel, x, y + 1), -params.smoothnessWeight);
                r[constraintIndex++] = 0.0f;
            }
        }
    }

    //
    //  Regularization constraints
    //
    for (int channel = 0; channel < 3; channel++)
    {
        for (int x = 0; x < imageDimension; x++)
        {
            for (int y = 0; y < imageDimension; y++)
            {
                f.insert((UINT)constraintIndex, mapPixelToArrray(channel, x, y), params.regularizationWeight);
                r[constraintIndex++] = imageNetMean(channel, x, y) * params.regularizationWeight;
            }
        }
    }

    cout << "solving system" << endl;

    ml::LinearSolverEigen<float> solver(ml::LinearSolverEigen<float>::ConjugateGradient_Diag, 0.01);

    //SparseMatrixf fT = f.transpose();

    //cout << "computing fTf" << endl;
    //SparseMatrixf fTf = f.transpose() * f;

    //auto p = solver.solve(fTf, fT * r);
    //auto p = solver.solveLeastSquaresNormalEquations(f, r);
    auto p = solver.solveLeastSquaresManualCG(f, r, 10000);
    
    
    reconstructedImage.allocate(imageDimension, imageDimension);
    for (auto &v : reconstructedImage)
    {
        v.value[0] = ml::util::boundToByte(p[0 * channelPitch + v.y * yPitch + v.x]);
        v.value[1] = ml::util::boundToByte(p[1 * channelPitch + v.y * yPitch + v.x]);
        v.value[2] = ml::util::boundToByte(p[2 * channelPitch + v.y * yPitch + v.x]);
    }
}

void CNN::filter(const Bitmap &input, LayerData &output) const
{
    LayerData transformedInput;
    transform.transform(input, transformedInput);
    layer.filter(transformedInput, output);
}

void CNN::initStandard()
{
    const string imageSynthDir = R"(\\etherion\share\data\imageSynth\)";
    transform.loadFromBlob(imageSynthDir + "imageNetMean.txt");
    layer.filters.loadFromBlob(imageSynthDir + "filterBank.txt", imageSynthDir + "bias.txt");
    //ml::LodePNG::save(layer.filters.makeVisualization(), R"(\\etherion\share\data\imageSynth\filterBank.png)");
}