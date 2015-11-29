#pragma once
#include <random>
#include <mutex>

#ifdef RNG_CPP
std::mutex RNGseedGeneratorMutex;
std::mt19937 RNGseedGenerator;
#else
extern std::mutex RNGseedGeneratorMutex;
extern std::mt19937 RNGseedGenerator;
#endif

class RNG {
  std::normal_distribution<> stdNormal;
  std::uniform_real_distribution<> uniform01;

public:
  std::mt19937 gen;

  RNG();
  int randint(int n);
  float uniform(float a = 0, float b = 1);
  float normal(float mean = 0, float sd = 1);
  int bernoulli(float p);
  template <typename T> int index(std::vector<T> &v);
  std::vector<int> NchooseM(int n, int m);
  std::vector<int> permutation(int n);
  template <typename T> void vectorShuffle(std::vector<T> &v);
};
