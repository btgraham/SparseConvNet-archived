class Picture {
public:
  virtual void codifyInputData (SpatiallySparseBatch &batch)=0;
  virtual Picture* distort (RNG& rng, batchType type) {return this;}
  virtual string identify() {return string();}
  int label; //-1 for unknown
  virtual ~Picture() {}
};

class SpatialDataset {
public:
  string name;
  vector<Picture*> pictures;
  int nFeatures;
  int nClasses;
  batchType type;
  void shuffle() {
    random_shuffle ( pictures.begin(), pictures.end());
  }
  void summary() {
    cout << "-------------------------------------------------------"<<endl;
    cout << "Name:           " << name << endl;
    cout << "nPictures:      " << pictures.size() << endl;
    cout << "nClasses:       " << nClasses << endl;
    cout << "nFeatures:      " << nFeatures << endl;
    cout << "Type:           " << batchTypeNames[type]<<endl;
    cout << "-------------------------------------------------------"<<endl;
  }
  SpatialDataset extractValidationSet(float p=0.1) {
    SpatialDataset val;
    val.name=name+string(" Validation set");
    name=name+string(" minus Validation set");
    val.nClasses=nClasses;
    val.nFeatures=nFeatures;
    val.type=TESTBATCH;
    shuffle();
    int size=pictures.size()*p;
    for (;size>0;size--) {
      val.pictures.push_back(pictures.back());
      pictures.pop_back();
    }
    return val;
  }
  void subsetOfClasses(vector<int> activeClasses) {
    nClasses=activeClasses.size();
    vector<Picture*> p=pictures;
    pictures.clear();
    for (int i=0;i<p.size();++i) {
      vector<int>::iterator it;
      it = find (activeClasses.begin(), activeClasses.end(), p[i]->label);
      if (it != activeClasses.end()) {
        p[i]->label=it-activeClasses.begin();
        pictures.push_back(p[i]);
        //cout << pictures.size() << " " << p[i]->identify() << endl;
      } else
        delete p[i];
    }
  }
  SpatialDataset subset(int n) {
    SpatialDataset subset(*this);
    subset.shuffle();
    subset.pictures.resize(n);
    return subset;
  }
};
