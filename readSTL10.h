//Implement index learning in SparseConvNet??
#include "STL-10.h"
void readSTL10fold(vector<Picture*> &characters, const char* filenameX, const char* filenameY, int* fold, int n) {
  ifstream fileX(filenameX,ios::in|ios::binary);
  ifstream fileY(filenameY,ios::in|ios::binary);
  if (!fileX) {
    cout <<"Cannot find " << filenameX << endl;
    exit(EXIT_FAILURE);
  }
  if (!fileY) {
    cout <<"Cannot find " << filenameY << endl;
    exit(EXIT_FAILURE);
  }
  for (int i=0;i<n;++i) {
    fileY.seekg(fold[i],fileY.beg);
    char label;
    fileY.read(&label,1);
    OpenCVPicture* character = new OpenCVPicture(96,96,3,128,label-1);
    fileX.seekg(fold[i]*96*96*3,fileX.beg);
    unsigned char bitmap[96*96*3];
    fileX.read((char*)bitmap,96*96*3);
    for (int x=0;x<96;x++) {
      for (int y=0;y<96;y++) {
        for (int c=0;c<3;c++) {
          character->mat.ptr()[x*96*3+y*3+(2-c)]=bitmap[c*96*96+y*96+x];
        }
      }
    }
    // writeImage(character->mat,i);cout<<"!\n";
    characters.push_back(character);
  }
  fileX.close();
  fileY.close();
}
SpatialDataset STL10TrainSet(int fold) {
  SpatialDataset dataset;
  dataset.name="STL-10 training fold "+ boost::lexical_cast<string>(fold);
  dataset.type=TRAINBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=10;
  char filenameX[]="Data/STL-10/train_X.bin";
  char filenameY[]="Data/STL-10/train_y.bin";
  readSTL10fold(dataset.pictures,filenameX,filenameY,STL10FoldIndices[fold],1000);
  return dataset;
}
SpatialDataset STL10TestSet() {
  SpatialDataset dataset;
  dataset.name="STL-10 test set";
  dataset.type=TESTBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=10;
  char filenameX[]="Data/STL-10/test_X.bin";
  char filenameY[]="Data/STL-10/test_y.bin";
  readSTL10fold(dataset.pictures,filenameX,filenameY,&STL10TestIndices[0],8000);
  return dataset;
}
