#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <chrono> // for time counting
#include <ctime> // time_t

using namespace std;
using namespace cv;
using namespace std::chrono; 

Point getCentermass(Mat &src);
void drawAxis(Mat&, Point, Point, Scalar, const float);
double getOrientation(const vector<Point> &, Mat&);

Point getCentermass(Mat &src){
    Moments m = moments(src,true);
    Point p(m.m10/m.m00, m.m01/m.m00);
    return p;
}
void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
    double angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    double hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, colour, 1, LINE_AA);
    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);
}
double getOrientation(Mat &img)
{
    //Construct a buffer used by the pca analysis
    vector<cv::Point2i> locations;  // output, locations of non-zero pixels 
    findNonZero(img, locations);
    //cout<<locations.at(2).x<<endl;
    int sz = locations.size();
    Mat data_pts = Mat(sz, 2, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = locations.at(i).x;
        data_pts.at<double>(i, 1) = locations.at(i).y;
    }
    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
    //Store the center of the object
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                      static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; i++)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    }
    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x)*180/M_PI; // orientation in radians
    return angle;
}
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
vector<int> getUniqueVector(Mat &mat){
    vector<int> v;
    for ( int i=0;i<mat.size[0];i++) {
        for ( int j=0;j<mat.size[1];j++) {
            v.push_back(int(mat.at<uchar>(i,j)));
        }
    }
    sort(v.begin(),v.end());
    v.erase(unique(v.begin(),v.end()),v.end());
    return v;
}
void printVector(vector<int> &v){
    for (int &x:v)
        cout << x << " ";
    cout << endl;
}
vector<int> getAvgWH(Mat &mat){
    vector<cv::Point2i> locations;  // output, locations of non-zero pixels 
    findNonZero(mat, locations);
    Rect rect = cv::boundingRect(locations);
    //cout<< rect.x<<"-"<<rect.y<<"-"<<rect.width<<"-"<<rect.height<<endl;
    vector<int> res = {rect.width,rect.height};
    return res;
}
vector<float> getRatio(Mat &mat0,Mat &mat1){
    vector<int> ratio0 = getAvgWH(mat0);
    vector<int> ratio1 = getAvgWH(mat1);
    vector<float> res = {float(ratio1.at(0))/float(ratio0.at(0)),
                        float(ratio1.at(1))/float(ratio0.at(1))};
    return res;
}
Mat rotateBW(Mat &img,float angle,Point center){
    Mat dst,dstBW;
    Mat M = getRotationMatrix2D(center, angle, 1.0);
    //cout << M.size()<< endl;
    warpAffine(img, dst, M, img.size());
    threshold(dst,dstBW,200,255,0);
    return dstBW;
}
void SFVMatchTemplate(const std::string &fn0,const std::string &fn1){
    Mat scaled_temp, saved;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    Mat rawimg0 = imread(fn0,0);
    Mat rawimg1 = imread(fn1,0);

    Point centermass0 = getCentermass(rawimg0);
    Point centermass1 = getCentermass(rawimg1);

    double angle0 = getOrientation(rawimg0);
    double angle1 = getOrientation(rawimg1);

    double d_angle = angle1 - angle0;
    Mat rotated = rotateBW(rawimg1,d_angle,centermass1);
    centermass1 = getCentermass(rotated);

    vector<float> scale = getRatio(rawimg0,rotated);
    Size newsize = {int(float(rawimg0.size[1])*scale.at(0)),
                            int(float(rawimg0.size[0])*scale.at(1))};
                 
    resize(rawimg0,scaled_temp,newsize,0,0,INTER_NEAREST);
    centermass0 = getCentermass(scaled_temp);

    Point offs = centermass1 - centermass0;
    findContours(scaled_temp, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    for(int i=0;i<contours.at(0).size();i++)
        contours.at(0).at(i) += offs;

    cvtColor(rotated, saved, COLOR_GRAY2BGR);
    drawContours(saved, contours, 0, Scalar(0, 0, 255), 2);
    imwrite("cpp_match.png",saved);
}
int main( int argc, char** argv ) {
    Mat src;
    int totcount = 50;

    auto start = steady_clock::now() ;

    for(int i=0;i<totcount;i++)
        SFVMatchTemplate("Binary_coins.png","1.png");

    auto end = steady_clock::now() ;

    double a = duration_cast<milliseconds>(end-start).count();
    cout << "CPP: " << a/totcount
              << " milliseconds / img\n" ;

    return EXIT_SUCCESS;
}
