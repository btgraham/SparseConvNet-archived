RNG::RNG() : nd(0,1), stdNormal(gen,nd), randomNumber(gen, uni_dist), random01(gen,uni_01) {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  RNGseedGeneratorMutex.lock();
  gen.seed(RNGseedGenerator()+ts.tv_nsec);
  RNGseedGeneratorMutex.unlock();
}
int RNG::randint(int n) {
  if (n==0) return 0;
  else return gen()%n;
}
float RNG::uniform(float a, float b) {
  return a+(b-a)*random01();
}
float RNG::normal(float mean, float sd) {
  return mean+sd*stdNormal();
}
int RNG::bernoulli(float p) {
  if (uniform()<p)
    return 1;
  else
    return 0;
}
template <typename T>
int RNG::index(std::vector<T> &v) {
  if (v.size()==0) std::cout << "RNG::index called for empty std::vector!\n";
  return gen()%v.size();
}
std::vector<int> RNG::NchooseM(int n, int m) {
  std::vector<int> ret(m,100);
  int ctr=m;
  for(int i=0;i<n;i++)
    if (uniform()<ctr*1.0/(n-i)) ret[m-ctr--]=i;
  return ret;
}
std::vector<int> RNG::permutation(int n) {
  std::vector<int> ret;
  for (int i=0;i<n;i++) ret.push_back(i);
  random_shuffle ( ret.begin(), ret.end(),randomNumber);
  return ret;
}
template <typename T> void RNG::vectorShuffle(std::vector<T> &v) {
  random_shuffle( v.begin(), v.end(), randomNumber);
}
