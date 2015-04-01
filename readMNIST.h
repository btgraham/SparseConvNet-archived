static int intToggleEndianness(int a) {
  int b=0;
  b+=a%256*(1<<24);a>>=8;
  b+=a%256*(1<<16);a>>=8;
  b+=a%256*(1<< 8);a>>=8;
  b+=a%256*(1<< 0);
  return b;}

static void loadMnistC(string filename, vector<Picture*> &characters) {
  ifstream f(filename.c_str());
  if (!f) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);}
  int a,n1,n2,n3;
  f.read((char*)&a,4);
  f.read((char*)&a,4);
  n1=intToggleEndianness(a);
  f.read((char*)&a,4);
  n2=intToggleEndianness(a);
  f.read((char*)&a,4);
  n3=intToggleEndianness(a);
  for (int i1=0;i1<n1;i1++) {
    OpenCVPicture* character = new OpenCVPicture(n2,n3,1,0);
    f.read((char *)character->mat.ptr(),n2*n3);
    characters.push_back(character);
  }
}

static void loadMnistL(string filename, vector<Picture*> &characters) {
  ifstream f(filename.c_str());
  if (!f) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);}
  int a,n;
  char l;
  f.read((char*)&a,4);
  f.read((char*)&a,4);
  n=intToggleEndianness(a);
  for (int i=0;i<n;i++) {
    f.read(&l,1);
    characters[i]->label=l;
  }
}

SpatialDataset MnistTrainSet() {
  SpatialDataset dataset;
  dataset.name="MNIST train set";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=1;
  dataset.nClasses=10;
  string trainC("Data/MNIST/train-images-idx3-ubyte");
  string trainL("Data/MNIST/train-labels-idx1-ubyte");
  loadMnistC(trainC, dataset.pictures);
  loadMnistL(trainL, dataset.pictures);
  return dataset;
}
SpatialDataset MnistTestSet() {
  SpatialDataset dataset;
  dataset.type=TESTBATCH;
  dataset.name="MNIST test set";
  dataset.nFeatures=1;
  dataset.nClasses=10;
  string testC("Data/MNIST/t10k-images-idx3-ubyte");
  string testL("Data/MNIST/t10k-labels-idx1-ubyte");
  loadMnistC(testC, dataset.pictures);
  loadMnistL(testL, dataset.pictures);
  return dataset;
}
