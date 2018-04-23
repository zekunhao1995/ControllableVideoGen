#ifndef DESCRIPTORS_H_
#define DESCRIPTORS_H_

#include "DenseTrackStab.h"
using namespace cv;

// get the rectangle for computing the descriptor
void GetRect(const Point2f& point, RectInfo& rect, const int width, const int height, const DescInfo& descInfo)
{
	int x_min = descInfo.width/2;
	int y_min = descInfo.height/2;
	int x_max = width - descInfo.width;
	int y_max = height - descInfo.height;

	rect.x = std::min<int>(std::max<int>(cvRound(point.x) - x_min, 0), x_max);
	rect.y = std::min<int>(std::max<int>(cvRound(point.y) - y_min, 0), y_max);
	rect.width = descInfo.width;
	rect.height = descInfo.height;
}

// compute integral histograms for the whole image
void BuildDescMat(const Mat& xComp, const Mat& yComp, float* desc, const DescInfo& descInfo)
{
	float maxAngle = 360.f;
	int nDims = descInfo.nBins;
	// one more bin for hof
	int nBins = descInfo.isHof ? descInfo.nBins-1 : descInfo.nBins;
	const float angleBase = float(nBins)/maxAngle;

	int step = (xComp.cols+1)*nDims;
	int index = step + nDims;
	for(int i = 0; i < xComp.rows; i++, index += nDims) {
		const float* xc = xComp.ptr<float>(i);
		const float* yc = yComp.ptr<float>(i);

		// summarization of the current line
		std::vector<float> sum(nDims);
		for(int j = 0; j < xComp.cols; j++) {
			float x = xc[j];
			float y = yc[j];
			float mag0 = sqrt(x*x + y*y);
			float mag1;
			int bin0, bin1;

			// for the zero bin of hof
			if(descInfo.isHof && mag0 <= min_flow) {
				bin0 = nBins; // the zero bin is the last one
				mag0 = 1.0;
				bin1 = 0;
				mag1 = 0;
			}
			else {
				float angle = fastAtan2(y, x);
				if(angle >= maxAngle) angle -= maxAngle;

				// split the mag to two adjacent bins
				float fbin = angle * angleBase;
				bin0 = cvFloor(fbin);
				bin1 = (bin0+1)%nBins;

				mag1 = (fbin - bin0)*mag0;
				mag0 -= mag1;
			}

			sum[bin0] += mag0;
			sum[bin1] += mag1;

			for(int m = 0; m < nDims; m++, index++)
				desc[index] = desc[index-step] + sum[m];
		}
	}
}

// get a descriptor from the integral histogram
void GetDesc(const DescMat* descMat, RectInfo& rect, DescInfo descInfo, std::vector<float>& desc, const int index)
{
	int dim = descInfo.dim;
	int nBins = descInfo.nBins;
	int height = descMat->height;
	int width = descMat->width;

	int xStride = rect.width/descInfo.nxCells;
	int yStride = rect.height/descInfo.nyCells;
	int xStep = xStride*nBins;
	int yStep = yStride*width*nBins;

	// iterate over different cells
	int iDesc = 0;
	std::vector<float> vec(dim);
	for(int xPos = rect.x, x = 0; x < descInfo.nxCells; xPos += xStride, x++)
	for(int yPos = rect.y, y = 0; y < descInfo.nyCells; yPos += yStride, y++) {
		// get the positions in the integral histogram
		const float* top_left = descMat->desc + (yPos*width + xPos)*nBins;
		const float* top_right = top_left + xStep;
		const float* bottom_left = top_left + yStep;
		const float* bottom_right = bottom_left + xStep;

		for(int i = 0; i < nBins; i++) {
			float sum = bottom_right[i] + top_left[i] - bottom_left[i] - top_right[i];
			vec[iDesc++] = std::max<float>(sum, 0) + epsilon;
		}
	}

	float norm = 0;
	for(int i = 0; i < dim; i++)
		norm += vec[i];
	if(norm > 0) norm = 1./norm;

	int pos = index*dim;
	for(int i = 0; i < dim; i++)
		desc[pos++] = sqrt(vec[i]*norm);
}

// for HOG descriptor
void HogComp(const Mat& img, float* desc, DescInfo& descInfo)
{
	Mat imgX, imgY;
	Sobel(img, imgX, CV_32F, 1, 0, 1);
	Sobel(img, imgY, CV_32F, 0, 1, 1);
	BuildDescMat(imgX, imgY, desc, descInfo);
}

// for HOF descriptor
void HofComp(const Mat& flow, float* desc, DescInfo& descInfo)
{
	Mat flows[2];
	split(flow, flows);
	BuildDescMat(flows[0], flows[1], desc, descInfo);
}

// for MBH descriptor
void MbhComp(const Mat& flow, float* descX, float* descY, DescInfo& descInfo)
{
	Mat flows[2];
	split(flow, flows);

	Mat flowXdX, flowXdY, flowYdX, flowYdY;
	Sobel(flows[0], flowXdX, CV_32F, 1, 0, 1);
	Sobel(flows[0], flowXdY, CV_32F, 0, 1, 1);
	Sobel(flows[1], flowYdX, CV_32F, 1, 0, 1);
	Sobel(flows[1], flowYdY, CV_32F, 0, 1, 1);

	BuildDescMat(flowXdX, flowXdY, descX, descInfo);
	BuildDescMat(flowYdX, flowYdY, descY, descInfo);
}

// check whether a trajectory is valid or not
bool IsValid(std::vector<Point2f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length)
{
	int size = track.size();
	float norm = 1./size;
	for(int i = 0; i < size; i++) {
		mean_x += track[i].x;
		mean_y += track[i].y;
	}
	mean_x *= norm;
	mean_y *= norm;

	for(int i = 0; i < size; i++) {
		float temp_x = track[i].x - mean_x;
		float temp_y = track[i].y - mean_y;
		var_x += temp_x*temp_x;
		var_y += temp_y*temp_y;
	}
	var_x *= norm;
	var_y *= norm;
	var_x = sqrt(var_x);
	var_y = sqrt(var_y);

	// remove static trajectory
	if(var_x < min_var && var_y < min_var)
		return false;
	// remove random trajectory
	if( var_x > max_var || var_y > max_var )
		return false;


    float fulllen = 0.;
	for(int i = 0; i < size-1; i++) {
		float dx = track[i+1].x - track[i].x;
    	float dy = track[i+1].y - track[i].y;
    	float temp = sqrt(dx*dx + dy*dy);
		
		fulllen += temp;
	}
	float cur_max = 0;
	float cur_min = 9999999;
	int bad_count = 0;
	float ddx = 0;
	float ddy = 0;
	float len2x = 0;
	float len2y = 0;
	for(int i = 0; i < size-1; i++) {
		//track[i] = track[i+1] - track[i];
		//float temp = sqrt(track[i].x*track[i].x + track[i].y*track[i].y);
		float dx = track[i+1].x - track[i].x;
    	float dy = track[i+1].y - track[i].y;
    	ddx += dx;
    	ddy += dy;
		float temp = sqrt(dx*dx + dy*dy);
		if(i >= 10) {
		    len2x += dx;
		    len2y += dy;
		}

		length += temp;
		if(temp > cur_max)
			cur_max = temp;
		if(temp < cur_min)
			cur_min = temp;
	    if(temp < fulllen*0.005)
	        bad_count++;
	}

	if(cur_max > max_dis && cur_max > length*0.7) // used to be 0.7
	    return false;
	if(bad_count > 5)
	    return false;
	
	//if(sqrt(ddx*ddx+ddy*ddy) < 4)
	//    return false;
	
	//if(sqrt(len2x*len2x+len2y*len2y) < 1)
	//    return false;
	    
	////if(cur_min < length*0.05 || cur_max > length*0.2) // used to be 0.7
	//if(cur_min < length*0.01)
	//    return false;
	//if(cur_max > length*0.3)
		

/*
	track.pop_back();
	norm = 1./length;
	// normalize the trajectory
	for(int i = 0; i < size-1; i++)
		track[i] *= norm;
*/
	return true;
}

bool IsCameraMotion(std::vector<Point2f>& disp)
{
	float disp_max = 0;
	float disp_sum = 0;
	for(int i = 0; i < disp.size(); ++i) {
		float x = disp[i].x;
		float y = disp[i].y;
		float temp = sqrt(x*x + y*y);

		disp_sum += temp;
		if(disp_max < temp)
			disp_max = temp;
	}

	//if(disp_max <= 5)
	if(disp_max <= 1)
		return false;

	float disp_norm = 1./disp_sum;
	for (int i = 0; i < disp.size(); ++i)
		disp[i] *= disp_norm;

	return true;
}

// detect new feature points in an image without overlapping to previous points
void DenseSample(const Mat& grey, std::vector<Point2f>& points, const double quality, const int min_distance)
{
	int width = grey.cols/min_distance;
	int height = grey.rows/min_distance;

	Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);

	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*quality;

	std::vector<int> counters(width*height);
	int x_max = min_distance*width;
	int y_max = min_distance*height;

	for(int i = 0; i < points.size(); i++) {
		Point2f point = points[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);

		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters[y*width+x]++;
	}

	points.clear();
	int index = 0;
	int offset = min_distance/2;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++, index++) {
		if(counters[index] > 0)
			continue;

		int x = j*min_distance+offset;
		int y = i*min_distance+offset;

		if(eig.at<float>(y, x) > threshold)
			points.push_back(Point2f(float(x), float(y)));
	}
}

void InitPry(const Mat& frame, std::vector<float>& scales, std::vector<Size>& sizes)
{
	int rows = frame.rows, cols = frame.cols;
	float min_size = std::min<int>(rows, cols);

	int nlayers = 0;
	while(min_size >= patch_size) {
		min_size /= scale_stride;
		nlayers++;
	}

	if(nlayers == 0) nlayers = 1; // at least 1 scale 

	scale_num = std::min<int>(scale_num, nlayers);

	scales.resize(scale_num);
	sizes.resize(scale_num);

	scales[0] = 1.;
	sizes[0] = Size(cols, rows);

	for(int i = 1; i < scale_num; i++) {
		scales[i] = scales[i-1] * scale_stride;
		sizes[i] = Size(cvRound(cols/scales[i]), cvRound(rows/scales[i]));
	}
}

void BuildPry(const std::vector<Size>& sizes, const int type, std::vector<Mat>& grey_pyr)
{
	int nlayers = sizes.size();
	grey_pyr.resize(nlayers);

	for(int i = 0; i < nlayers; i++)
		grey_pyr[i].create(sizes[i], type);
}

void DrawTrack(const std::vector<Point2f>& point, const int index, const float scale, Mat& image)
{
	Point2f point0 = point[0];
	point0 *= scale;

	for (int j = 1; j <= index; j++) {
		Point2f point1 = point[j];
		point1 *= scale;

		line(image, point0, point1, Scalar(0,cvFloor(255.0*(j+1.0)/float(index+1.0)),0), 2, 8, 0);
		point0 = point1;
	}
	circle(image, point0, 2, Scalar(0,0,255), -1, 8, 0);
}

void PrintDesc(std::vector<float>& desc, DescInfo& descInfo, TrackInfo& trackInfo)
{
	int tStride = cvFloor(trackInfo.length/descInfo.ntCells);
	float norm = 1./float(tStride);
	int dim = descInfo.dim;
	int pos = 0;
	for(int i = 0; i < descInfo.ntCells; i++) {
		std::vector<float> vec(dim);
		for(int t = 0; t < tStride; t++)
			for(int j = 0; j < dim; j++)
				vec[j] += desc[pos++];
		for(int j = 0; j < dim; j++)
			printf("%.7f\t", vec[j]*norm);
	}
}

void LoadBoundBox(char* file, std::vector<Frame>& bb_list)
{
	// load the bouding box file
    std::ifstream bbFile(file);
    std::string line;

    while(std::getline(bbFile, line)) {
		 std::istringstream iss(line);

		int frameID;
		if (!(iss >> frameID))
			continue;

		Frame cur_frame(frameID);

		float temp;
		std::vector<float> a(0);
		while(iss >> temp)
			a.push_back(temp);

		int size = a.size();

		if(size % 5 != 0)
			fprintf(stderr, "Input bounding box format wrong!\n");

		for(int i = 0; i < size/5; i++)
			cur_frame.BBs.push_back(BoundBox(a[i*5], a[i*5+1], a[i*5+2], a[i*5+3], a[i*5+4]));

		bb_list.push_back(cur_frame);
    }
}

void InitMaskWithBox(Mat& mask, std::vector<BoundBox>& bbs)
{
	int width = mask.cols;
	int height = mask.rows;

	for(int i = 0; i < height; i++) {
		uchar* m = mask.ptr<uchar>(i);
		for(int j = 0; j < width; j++)
			m[j] = 1;
	}

	for(int k = 0; k < bbs.size(); k++) {
		BoundBox& bb = bbs[k];
		for(int i = cvCeil(bb.TopLeft.y); i <= cvFloor(bb.BottomRight.y); i++) {
			uchar* m = mask.ptr<uchar>(i);
			for(int j = cvCeil(bb.TopLeft.x); j <= cvFloor(bb.BottomRight.x); j++)
				m[j] = 0;
		}
	}
}

static void MyWarpPerspective(Mat& prev_src, Mat& src, Mat& dst, Mat& M0, int flags = INTER_LINEAR,
	            			 int borderType = BORDER_CONSTANT, const Scalar& borderValue = Scalar())
{
	int width = src.cols;
	int height = src.rows;
	dst.create( height, width, CV_8UC1 );

	Mat mask = Mat::zeros(height, width, CV_8UC1);
	const int margin = 5;

    const int BLOCK_SZ = 32;
    short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];

    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    double M[9];
    Mat matM(3, 3, CV_64F, M);
    M0.convertTo(matM, matM.type());
    if( !(flags & WARP_INVERSE_MAP) )
         invert(matM, matM);

    int x, y, x1, y1;

    int bh0 = std::min(BLOCK_SZ/2, height);
    int bw0 = std::min(BLOCK_SZ*BLOCK_SZ/bh0, width);
    bh0 = std::min(BLOCK_SZ*BLOCK_SZ/bw0, height);

    for( y = 0; y < height; y += bh0 ) {
    for( x = 0; x < width; x += bw0 ) {
		int bw = std::min( bw0, width - x);
        int bh = std::min( bh0, height - y);

        Mat _XY(bh, bw, CV_16SC2, XY);
		Mat matA;
        Mat dpart(dst, Rect(x, y, bw, bh));

		for( y1 = 0; y1 < bh; y1++ ) {

			short* xy = XY + y1*bw*2;
            double X0 = M[0]*x + M[1]*(y + y1) + M[2];
            double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
            double W0 = M[6]*x + M[7]*(y + y1) + M[8];
            short* alpha = A + y1*bw;

            for( x1 = 0; x1 < bw; x1++ ) {

                double W = W0 + M[6]*x1;
                W = W ? INTER_TAB_SIZE/W : 0;
                double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0]*x1)*W));
                double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3]*x1)*W));
 
				double _X = fX/double(INTER_TAB_SIZE);
				double _Y = fY/double(INTER_TAB_SIZE);

				if( _X > margin && _X < width-1-margin && _Y > margin && _Y < height-1-margin )
					mask.at<uchar>(y+y1, x+x1) = 1;

                int X = saturate_cast<int>(fX);
                int Y = saturate_cast<int>(fY);

                xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
                xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
                alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (X & (INTER_TAB_SIZE-1)));
            }
        }

        Mat _matA(bh, bw, CV_16U, A);
        remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
    }
    }

	for( y = 0; y < height; y++ ) {
		const uchar* m = mask.ptr<uchar>(y);
		const uchar* s = prev_src.ptr<uchar>(y);
		uchar* d = dst.ptr<uchar>(y);
		for( x = 0; x < width; x++ ) {
			if(m[x] == 0)
				d[x] = s[x];
		}
	}
}

cv::Mat windowedMatchingMask( const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
                          float maxDeltaX, float maxDeltaY )
{
  if( keypoints1.empty() || keypoints2.empty() )
    return cv::Mat();

  int n1 = (int)keypoints1.size(), n2 = (int)keypoints2.size();
  cv::Mat mask( n1, n2, CV_8UC1 );
  for( int i = 0; i < n1; i++ )
    {
      for( int j = 0; j < n2; j++ )
        {
          cv::Point2f diff = keypoints2[j].pt - keypoints1[i].pt;
          mask.at<uchar>(i, j) = std::abs(diff.x) < maxDeltaX && std::abs(diff.y) < maxDeltaY;
        }
    }
  return mask;
}

void ComputeMatch(const std::vector<KeyPoint>& prev_kpts, const std::vector<KeyPoint>& kpts,
				  const Mat& prev_desc, const Mat& desc, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts)
{
	prev_pts.clear();
	pts.clear();

	if(prev_kpts.size() == 0 || kpts.size() == 0)
		return;

	Mat mask = windowedMatchingMask(kpts, prev_kpts, 25, 25);

	BFMatcher desc_matcher(NORM_L2);
	std::vector<DMatch> matches;

	desc_matcher.match(desc, prev_desc, matches, mask);
	
	prev_pts.reserve(matches.size());
	pts.reserve(matches.size());

	for(size_t i = 0; i < matches.size(); i++) {
		const DMatch& dmatch = matches[i];
		// get the point pairs that are successfully matched
		prev_pts.push_back(prev_kpts[dmatch.trainIdx].pt);
		pts.push_back(kpts[dmatch.queryIdx].pt);
	}

	return;
}

void MergeMatch(const std::vector<Point2f>& prev_pts1, const std::vector<Point2f>& pts1,
				const std::vector<Point2f>& prev_pts2, const std::vector<Point2f>& pts2,
				std::vector<Point2f>& prev_pts_all, std::vector<Point2f>& pts_all)
{
	prev_pts_all.clear();
	prev_pts_all.reserve(prev_pts1.size() + prev_pts2.size());

	pts_all.clear();
	pts_all.reserve(pts1.size() + pts2.size());

	for(size_t i = 0; i < prev_pts1.size(); i++) {
		prev_pts_all.push_back(prev_pts1[i]);
		pts_all.push_back(pts1[i]);
	}

	for(size_t i = 0; i < prev_pts2.size(); i++) {
		prev_pts_all.push_back(prev_pts2[i]);
		pts_all.push_back(pts2[i]);	
	}

	return;
}

void MatchFromFlow(const Mat& prev_grey, const Mat& flow, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts, const Mat& mask)
{
	int width = prev_grey.cols;
	int height = prev_grey.rows;
	prev_pts.clear();
	pts.clear();

	const int MAX_COUNT = 1000;
	goodFeaturesToTrack(prev_grey, prev_pts, MAX_COUNT, 0.001, 3, mask);
	
	if(prev_pts.size() == 0)
		return;

	for(int i = 0; i < prev_pts.size(); i++) {
		int x = std::min<int>(std::max<int>(cvRound(prev_pts[i].x), 0), width-1);
		int y = std::min<int>(std::max<int>(cvRound(prev_pts[i].y), 0), height-1);

		const float* f = flow.ptr<float>(y);
		pts.push_back(Point2f(x+f[2*x], y+f[2*x+1]));
	}
}

#endif /*DESCRIPTORS_H_*/
