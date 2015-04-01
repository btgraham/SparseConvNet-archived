class FloatPoint {
public:
  float x;
  float y;
  FloatPoint() {}
  FloatPoint(float x, float y) : x(x), y(y) {}
  void flip_horizontal() {
    x=-x;
  }
  void stretch_x(float alpha) {
    x*=(1+alpha);
  }
  void stretch_y(float alpha) {
    y*=(1+alpha);
  }
  void rotate(float angle) {
    float c=cos(angle);
    float s=sin(angle);
    float xx=+x*c+y*s;
    float yy=-x*s+y*c;
    x=xx;
    y=yy;
  }
  void slant_x(float alpha) {
    y+=alpha*x;
  }
  void slant_y(float alpha) {
    x+=alpha*y;
  }
  void stretch4(float cxx, float cxy, float cyx, float cyy) {
    float tx=x;
    float ty=y;
    x=(1+cxx)*tx+cxy*ty;
    y=(1+cyy)*ty+cyx*tx;
  }
};

typedef vector<FloatPoint> OnlinePenStroke;
typedef vector<OnlinePenStroke> OnlinePenStrokes;

void stretchXY(OnlinePenStrokes &character, RNG &rng, float max_stretch)
{
  float dx=rng.uniform(-max_stretch,max_stretch);
  float dy=rng.uniform(-max_stretch,max_stretch);
  for (int i=0;i<character.size();i++) {
    for (int j=0;j<character[i].size();j++) {
      character[i][j].stretch_x(dx);
      character[i][j].stretch_y(dy);
    }
  }
}

void rotate(OnlinePenStrokes &character, RNG &rng, float max_angle) {
  float angle=rng.uniform(-max_angle,max_angle);
  for (int i=0;i<character.size();i++) {
    for (int j=0;j<character[i].size();j++) {
      character[i][j].rotate(angle);
    }
  }
}
void slant_x(OnlinePenStrokes &character, RNG &rng, float max_alpha) {
  float alpha=rng.uniform(-max_alpha,max_alpha);
  for (int i=0;i<character.size();i++)
    for (int j=0;j<character[i].size();j++)
      character[i][j].slant_x(alpha);
}
void slant_y(OnlinePenStrokes &character, RNG &rng, float max_alpha) {
  float alpha=rng.uniform(-max_alpha,max_alpha);
  for (int i=0;i<character.size();i++)
    for (int j=0;j<character[i].size();j++)
      character[i][j].slant_y(alpha);
}
void stretch4(OnlinePenStrokes &character, RNG &rng, float max_stretch) {
  float cxx=rng.uniform(-max_stretch,+max_stretch);
  float cxy=rng.uniform(-max_stretch,+max_stretch);
  float cyx=rng.uniform(-max_stretch,+max_stretch);
  float cyy=rng.uniform(-max_stretch,+max_stretch);
  for (int i=0;i<character.size();i++)
    for (int j=0;j<character[i].size();j++)
      character[i][j].stretch4(cxx,cxy,cyx,cyy);
}

void jiggleCharacter(OnlinePenStrokes &character, RNG &rng, float max_delta) {
  float dx=rng.uniform(-max_delta,max_delta);
  float dy=rng.uniform(-max_delta,max_delta);
  for (int i=0;i<character.size();i++)
    for (int j=0;j<character[i].size();j++) {
      character[i][j].x+=dx;
      character[i][j].y+=dy;
    }
}
void jiggleStrokes(OnlinePenStrokes &character, RNG &rng, float max_delta) {
  for (int i=0;i<character.size();i++) {
    float dx=rng.uniform(-max_delta,max_delta);
    float dy=rng.uniform(-max_delta,max_delta);
    for (int j=0;j<character[i].size();j++) {
      character[i][j].x+=dx;
      character[i][j].y+=dy;
    }
  }
}

// Fit characters inside an
//  onlineHandwritingCharacterScale x onlineHandwritingCharacterScale box,
//  center the origin
void normalize(OnlinePenStrokes &ops)
{
  float x=ops[0][0].x;
  float X=ops[0][0].x;
  float y=ops[0][0].y;
  float Y=ops[0][0].y;
  for (int i=0;i<ops.size();i++) {
    for (int j=0;j<ops[i].size();j++) {
      x=min(ops[i][j].x,x);
      X=max(ops[i][j].x,X);
      y=min(ops[i][j].y,y);
      Y=max(ops[i][j].y,Y);
    }
  }
  float scaleF=onlineHandwritingCharacterScale/max(max(X-x,Y-y),0.0001f);
  for (int i=0;i<ops.size();i++) {
    for (int j=0;j<ops[i].size();j++) {
      ops[i][j].x=(ops[i][j].x-0.5*(X+x))*scaleF;
      ops[i][j].y=(ops[i][j].y-0.5*(Y+y))*scaleF;
    }
  }
}


int mapToGrid(float coord, int scale_N) {
  return max(0,min(scale_N-1,(int)(coord+0.5*scale_N)));
}

OnlinePenStroke constantSpeed(OnlinePenStroke &path, float density, int multiplier = 1) {
  vector<float> lengths(path.size());
  lengths[0]=0;
  for (int i=1;i<path.size();i++) {
    lengths[i]=lengths[i-1]+pow(pow(path[i].x-path[i-1].x,2)+pow(path[i].y-path[i-1].y,2),0.5);
  }
  float lTotal=lengths[path.size()-1];
  int n=(int)(0.5+lTotal/density);
  n*=multiplier;
  OnlinePenStroke r(n+1);
  int j=0;
  float alpha;
  r[0].x=path[0].x;
  r[0].y=path[0].y;
  for (int i=1;i<=n;i++) {
    while(n*lengths[j+1]<i*lTotal) j++;
    alpha=(lengths[j+1]-i*lTotal/n)/(lengths[j+1]-lengths[j]);
    r[i].x=path[j].x*alpha+path[j+1].x*(1-alpha);
    r[i].y=path[j].y*alpha+path[j+1].y*(1-alpha);
  }
  return r;
}

class OnlinePicture : public Picture {
public:
  OnlinePenStrokes ops;
  void codifyInputData (SpatiallySparseBatch &batch);
  Picture* distort (RNG& rng, batchType type=TRAINBATCH);
  OnlinePicture(int label_ = -1) {label=label_;}
  ~OnlinePicture() {}
};



//Example distortion functions

// Picture* OnlinePicture::distort(RNG& rng) {
//   OnlinePicture* pic=new OnlinePicture(*this);
//   jiggleStrokes(pic->ops,rng,1);
//   stretchXY(pic->ops,rng,0.3);
//   int r=rng.randint(3);
//   if (r==0) rotate(pic->ops,rng,0.3);
//   if (r==1) slant_x(pic->ops,rng,0.3);
//   if (r==2) slant_y(pic->ops,rng,0.3);
//   jiggleCharacter(pic->ops,rng,12);
//   return pic;
// }


// EDfields edf(pow(10,5),24,28,6,3);
// Picture* OnlinePicture::distort(RNG& rng) {
//   OnlinePicture* pic=new OnlinePicture(*this);
//   int ind=rng.index(edf.edf);
//   characterED(pic->ops,edf.edf[ind]);
//   return pic;
// }


vector<float> onlineHandwritingRegularizingConstants(nInputFeatures,1);
//Size nInputFeatures -- set to all ones to calculate appropriate values.

void calculateRegularizingConstants(SpatialDataset &dataset) {
  cout << "Using " << dataset.pictures.size() << " training samples to calculate regularizing constants." << endl;
  SpatiallySparseBatch batch(UNLABELLEDBATCH, nInputFeatures, onlineHandwritingCharacterScale+10,1);

  for (int i=0;i<dataset.pictures.size() and i<10000 and batch.interfaces[0].nSpatialSites<pow(10,7);++i)
    dataset.pictures[i]->codifyInputData(batch);
  for (int i=0; i<nInputFeatures; i++) {
    onlineHandwritingRegularizingConstants[i]=0;
    for (int j=0; j<batch.interfaces[0].nSpatialSites; j++)
      onlineHandwritingRegularizingConstants[i]=
        max(abs(batch.interfaces[0].features.hVector()[i+j*nInputFeatures]),
            onlineHandwritingRegularizingConstants[i]);
  }
  cout << "Regularizing constants: ";
  for (int i=0; i<nInputFeatures; i++)
    cout << onlineHandwritingRegularizingConstants[i] << " ";
  cout << endl;
}






















// // convert +append assamese*ppm assamese.png && rm assamese*ppm
// vector<int> drawPPM_counter(nCharacters);
// void drawPPM(OnlinePenStrokes &paths,int label) {
//   string filename=string("assamese_")+boost::lexical_cast<string>(label)+string("_")+boost::lexical_cast<string>(drawPPM_counter[label]++)+string(".ppm");
//   cout << endl<<filename << endl;
//   ofstream f(filename.c_str());
//   f << "P2\n"<< scale_N << " " << scale_N<< endl<< 1 << endl;

//   vector<int> grid(scale_N*scale_N,1);
//   for (int i=0; i<paths.size(); i++) {
//     OnlinePenStroke csp=constantSpeed(paths[i],3.0,6);
//     for (int j=0; j<csp.size(); j++) {
//       int n=mapToGrid(csp[j].x,scale_N)*scale_N+mapToGrid(csp[j].y,scale_N);
//       grid[n]=0;
//     }
//   }
//   //for (int x=scale_N-1; x>=0;x--) {
//   for (int x=0;x<scale_N; x++) {
//     for (int y=0; y<scale_N; y++) {
//       f << grid[x*scale_N+y] << " ";
//     }
//     f << endl;
//   }
//   f.close();
// }


// void drawGraphs(OnlinePenStrokes &paths, OnlinePenStrokes &paths2) {
//   vector<int> g(2*scale_N*scale_N,0);
//   for (int i=0; i<paths.size(); i++) {
//     OnlinePenStroke csp=constantSpeed(paths[i],3.0,6);
//     for (int j=0; j<csp.size(); j++) {
//       int n=mapToGrid(csp[j].x,scale_N)*2*scale_N+mapToGrid(csp[j].y,scale_N);
//       g[n]=1;
//     }
//   }
//   for (int i=0; i<paths2.size(); i++) {
//     OnlinePenStroke csp=constantSpeed(paths2[i],3.0,6);
//     for (int j=0; j<csp.size(); j++) {
//       int n=mapToGrid(csp[j].x,scale_N)*2*scale_N+mapToGrid(csp[j].y,scale_N)+scale_N;
//       g[n]=1;
//     }
//   }
//   for(int i=0; i< scale_N+2;i++) cout <<"--";cout<<endl;
//   for(int i=0; i<scale_N; i++) {
//     cout <<".";
//     for(int j=0; j<2*scale_N; j++) {
//       if (g[i*2*scale_N+j]==0)
//         cout << " ";
//       else
//         cout << "X";
//     }
//     cout <<"."<< endl;
//   }
//   for(int i=0; i< scale_N+2;i++) cout <<"--";cout<<endl;
// }

// void show_characters() {
//   RNG rng;
//   OnlinePenStroke l;
//   {
//     FloatPoint p;
//     float sn=0.5*(scale_n+1);
//     p.x=-sn;
//     for (p.y=-sn;p.y<sn;p.y++)
//       l.push_back(p);
//     p.y=sn;
//     for (p.x=-sn;p.x<sn;p.x++)
//       l.push_back(p);
//     p.x=sn;
//     for (p.y=sn;p.y>-sn;p.y--)
//       l.push_back(p);
//     p.y=-sn;
//     for (p.x=sn;p.x>-sn;p.x--)
//       l.push_back(p);
//   }
//   while(true) {
//     int i=rng.index(trainCharacters);
//     OnlinePicture* a=new OnlinePicture(*dynamic_cast<OnlinePicture*>(trainCharacters[i]));
//     a->ops.push_back(l);
//     OnlinePicture* b=dynamic_cast<OnlinePicture*>(a->distort(rng));
//     cout << i << " " << a->label<<endl;
//     printOnlinePenStrokes(a->ops);
//     drawGraphs(a->ops, b->ops);
//     delete a, b;
//     sleep(4);
//   }
// }
