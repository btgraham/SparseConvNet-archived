#pragma once

boost::mutex RNGseedGeneratorMutex;
boost::mt19937 RNGseedGenerator;

class RNG {
  boost::normal_distribution<> nd;
  boost::uniform_int<> uni_dist;
  boost::uniform_01<> uni_01;
public:
  boost::mt19937 gen;
  boost::variate_generator<boost::mt19937&,boost::normal_distribution<> > stdNormal;
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > randomNumber;
  boost::variate_generator<boost::mt19937&, boost::uniform_01<> > random01;

  RNG();
  int randint(int n);
  float uniform(float a=0, float b=1);
  float normal(float mean=0, float sd=1);
  int bernoulli(float p);
  template <typename T> int index(std::vector<T> &v);
  std::vector<int> NchooseM(int n, int m);
  std::vector<int> permutation(int n);
  template <typename T> void vectorShuffle(std::vector<T> &v);
};
