class SpatiallySparseLayer {
public:
  virtual void preprocess
  (SpatiallySparseBatchInterface &input, SpatiallySparseBatchInterface &output) = 0;
  virtual void forwards
  (SpatiallySparseBatchInterface &input, SpatiallySparseBatchInterface &output) = 0;
  virtual void scaleWeights
  (SpatiallySparseBatchInterface &input, SpatiallySparseBatchInterface &output) {};
  virtual void backwards
  (SpatiallySparseBatchInterface &input, SpatiallySparseBatchInterface &output, float learningRate) = 0;
  virtual void loadWeightsFromStream(ifstream &f) {};
  virtual void putWeightsToStream(ofstream &f)  {};
  virtual int calculateInputSpatialSize(int outputSpatialSize) = 0;
};
