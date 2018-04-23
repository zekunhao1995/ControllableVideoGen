#ifndef DENSETRACKSTAB_H_
#define DENSETRACKSTAB_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <ctype.h>
#include <unistd.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core/core.hpp"
//#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;

typedef struct 
{
    int traj_length;
    int num_trajs;
    float* out_trajs;
} Ret;

extern "C" void free_mem();
extern "C" void main_like(char* in_video, int in_width, int in_height, int in_frames, Ret* ret);

int start_frame = 0;
int end_frame = INT_MAX;
int scale_num = 8;
const float scale_stride = sqrt(2);
char* bb_file = NULL;

// parameters for descriptors
int patch_size = 32;
int nxy_cell = 2;
int nt_cell = 3;
float epsilon = 0.05;
const float min_flow = 0.4;

// parameters for tracking
double quality = 0.001;
int min_distance = 5;
int init_gap = 1;
int track_length = 15;

// parameters for rejecting trajectory
const float min_var = sqrt(3);
const float max_var = 50;
const float max_dis = 20;

typedef struct {
	int x;       // top left corner
	int y;
	int width;
	int height;
}RectInfo;

typedef struct {
    int width;   // resolution of the video
    int height;
    int length;  // number of frames
}SeqInfo;

typedef struct {
    int length;  // length of the trajectory
    int gap;     // initialization gap for feature re-sampling 
}TrackInfo;

typedef struct {
    int nBins;   // number of bins for vector quantization
    bool isHof; 
    int nxCells; // number of cells in x direction
    int nyCells; 
    int ntCells;
    int dim;     // dimension of the descriptor
    int height;  // size of the block for computing the descriptor
    int width;
}DescInfo; 

// integral histogram for the descriptors
typedef struct {
    int height;
    int width;
    int nBins;
    float* desc;
}DescMat;

class Track
{
public:
    std::vector<Point2f> point;
    std::vector<Point2f> disp;
    std::vector<float> hog;
    std::vector<float> hof;
    std::vector<float> mbhX;
    std::vector<float> mbhY;
    int index;

    Track(const Point2f& point_, const TrackInfo& trackInfo, const DescInfo& hogInfo,
          const DescInfo& hofInfo, const DescInfo& mbhInfo)
        : point(trackInfo.length+1), disp(trackInfo.length), hog(hogInfo.dim*trackInfo.length),
          hof(hofInfo.dim*trackInfo.length), mbhX(mbhInfo.dim*trackInfo.length), mbhY(mbhInfo.dim*trackInfo.length)
    {
        index = 0;
        point[0] = point_;
    }

    void addPoint(const Point2f& point_)
    {
        index++;
        point[index] = point_;
    }
};

class BoundBox
{
public:
	Point2f TopLeft;
	Point2f BottomRight;
	float confidence;

	BoundBox(float a1, float a2, float a3, float a4, float a5)
	{
		TopLeft.x = a1;
		TopLeft.y = a2;
		BottomRight.x = a3;
		BottomRight.y = a4;
		confidence = a5;
	}
};

class Frame
{
public:
	int frameID;
	std::vector<BoundBox> BBs;
	
	Frame(const int& frame_)
	{
		frameID = frame_;
		BBs.clear();
	}
};

#endif /*DENSETRACKSTAB_H_*/
