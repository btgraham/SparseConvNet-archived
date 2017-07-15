#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Rng.h"

void readImage(std::string filename, cv::Mat &mat, int flags = 1);
void writeImage(cv::Mat &mat, int n);
void writeImage(cv::Mat &mat, std::string filename);
void transformImage(cv::Mat &mat, int backgroundColor, float c00, float c01,
                    float c10, float c11);
void cropImage(cv::Mat &src, int X, int Y, int Width, int Height);
void distortImageColor(cv::Mat &mat, RNG &rng, int backgroundColor,
                       float sigma1, float sigma2, float sigma3, float sigma4);
