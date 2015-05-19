#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void readImage(std::string filename, cv::Mat& mat, int flags=1);
void writeImage(cv::Mat& mat, int n);
void writeImage(cv::Mat& mat, std::string filename);
void readTransformedImage(std::string filename, cv::Mat& dst, float scale,
                          float c00=1,float c01=0,float c10=0,float c11=1,
                          int backgroundColor=128,
                          int x=0, int X=-1, int y=0, int Y=-1);
void transformImage(cv::Mat &mat, int backgroundColor,float c00,float c01,float c10,float c11);
void cropImage(cv::Mat& src, int X, int Y, int Width, int Height);
