
struct ReconstructionParams
{
    ReconstructionParams();

    string toString() const
    {
        auto format = [](float v) -> string {
            string s = to_string(v);
            if (s.size() > 4)
                s.resize(4);
            return s;
        };
        return "_q" + format(quartile) + "_s" + format(smoothnessWeight) + "_r" + format(regularizationWeight);
    }

    float quartile;
    float smoothnessWeight;
    float regularizationWeight;
};

struct LayerImage
{
    LayerImage() {}
    LayerImage(const Grid2<float> &g)
    {
        values = g;
    }
    Bitmap makeVisualization() const;
    Bitmap makeVisualization(const ReconstructionParams &params) const;
    float comptueCutoff(const ReconstructionParams &params) const;

    Grid2<float> values;
};

struct LayerData
{
    LayerData() {}
    LayerData(const Bitmap &bmp)
    {
        images.push_back(LayerImage(Grid2<float>(bmp.getWidth(), bmp.getHeight(), [&](size_t x, size_t y) { return bmp(x, y).r / 255.0f; })));
        images.push_back(LayerImage(Grid2<float>(bmp.getWidth(), bmp.getHeight(), [&](size_t x, size_t y) { return bmp(x, y).g / 255.0f; })));
        images.push_back(LayerImage(Grid2<float>(bmp.getWidth(), bmp.getHeight(), [&](size_t x, size_t y) { return bmp(x, y).b / 255.0f; })));
    }

    vector<LayerImage> images;
};

struct Filter
{
    float filter(const LayerData &layer, int xStart, int yStart) const;
    vector< pair<UINT, float> > makeFilterRow(int xStart, int yStart, int yPitch, int channelPitch) const;

    //
    // 3x11x11 for the 1st layer, accessed as values(3,11,11)
    //
    Bitmap makeVisualization() const;

    float bias;
    Grid3<float> values;
};

struct FilterBank
{
    void loadFromBlob(const string &filterBlobFile, const string &biasBlobFile);
    Bitmap makeVisualization() const;

    UINT stride;
    vector<Filter> filters;
};

struct DataTransform
{
    void loadFromBlob(const string &blobFile);
    void transform(const Bitmap &input, LayerData &output) const;

    Grid3<float> meanValues;
};

class CNNLayer
{
public:
    void filter(const LayerData &input, LayerData &output) const;
    void invert(const ReconstructionParams &params, const LayerData &nextLayer, const Grid3<float> &imageNetMean, Bitmap &reconstructedImage) const;

    FilterBank filters;
};

class CNN
{
public:
    void initStandard();
    void filter(const Bitmap &input, LayerData &output) const;

    DataTransform transform;
    CNNLayer layer;
};