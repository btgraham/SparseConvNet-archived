//Batches

enum batchType {TRAINBATCH, TESTBATCH, UNLABELLEDBATCH, RESCALEBATCH};
const char *batchTypeNames[] ={ "TRAINBATCH", "TESTBATCH", "UNLABELLEDBATCH","RESCALEBATCH"};

class SpatiallySparseBatchInterface {
public:
  batchType type;
  int batchSize;                           // Number of training/test images
  int nFeatures;                           // Features per spatial location
  // Not dropped out features per spatial location
  vectorCUDA<int> featuresPresent;         // For dropout rng.NchooseM(nFeatures,featuresPresent.size());
  int spatialSize;                         // spatialSize x spatialSize grid
  int nSpatialSites;                       // Total active spatial locations within the
  //                                          batchSize x spatialSize x spatialSize
  //                                          possible locations.
  vectorCUDA<float> features;              // For the forwards pass
  vectorCUDA<float> dfeatures;             // For the backwards/backpropagation pass
  bool backpropErrors;                     // Calculate dfeatures? (false until after the first NiN layer)
  vector<vector<int> > grids;              // batchSize vectors of size spatialSize x spatialSize
  //                                          Store locations of nSpatialSites in the
  //                                          spatialSize x spatialSize grids
  vector<int> backgroundNullVectorNumbers; // Length batchSize, entry == -1 if corresponding
  //                                          grid has no 'null' vectors
  // Below used internally for convolution/pooling operation:
  vectorCUDA<int> poolingChoices;
  vectorCUDA<int> rules;
  void summary() {
    cout << "---------------------------------------------------\n";
    cout << "type" << type << endl;
    cout << "batchSize" << batchSize << endl;
    cout << "nFeatures" << nFeatures << endl;
    cout << "featuresPresent.size()" << featuresPresent.size() <<endl;
    cout << "spatialSize" << spatialSize << endl;
    cout << "nSpatialSites" << nSpatialSites << endl;
    cout << "features.size()" << features.size() << endl;
    cout << "dfeatures.size()" << dfeatures.size() << endl;
    cout << "grids.size()" << grids.size() << endl;
    cout << "grids[0].size()" << grids[0].size() << endl;
    cout << "backgroundNullVectorNumbers.size()" << backgroundNullVectorNumbers.size() << endl;
    cout << "type " << batchTypeNames[type]<<endl;
    cout << "---------------------------------------------------\n";
  }
};

class SpatiallySparseBatch {
public:
  vector<int> sampleNumbers;
  vector<SpatiallySparseBatchInterface> interfaces;
  vectorCUDA<int> labels;
  vector<vector<int> > predictions;
  vector<vector<float> > probabilities;
  float negativeLogLikelihood;
  int mistakes;
  SpatiallySparseBatch(batchType type,int nFeatures,int spatialSize, int nInterfaces) {
    mistakes=0;
    negativeLogLikelihood=0;
    interfaces.resize(nInterfaces);
    interfaces[0].type=type;
    interfaces[0].nFeatures=nFeatures;
    interfaces[0].spatialSize=spatialSize;
    interfaces[0].nSpatialSites=0;
    interfaces[0].featuresPresent.hVector()=range(nFeatures);
    interfaces[0].batchSize=0;
    interfaces[0].backpropErrors=false;
  }
};
