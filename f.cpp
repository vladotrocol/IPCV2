//Detect dartboards in various images

#include <stdio.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace std;

/** Function Headers */

//---------Main-------------
void detectAndDisplay(Mat frame);

//---------Detectors--------------
void haarDetect(Mat frame);
void houghCircles(Mat frame);
void houghLines(Mat frame);
void surfDetect(Mat frame);
void detectContours(Mat frame);

//--------Helpers-----------------
Mat add_matrix(Mat m1, Mat m2, float w1, float w2);
string int_to_string(int i);
Mat genGauss(float sigma, int gaussSize);
vector<Point2f> convert_faces_to_points(void);
vector<Point2f> convert_circles_to_points(void);
vector<Point2f> compute_intersection_points(void);
Mat assign_gauss(vector<Point2f> points, int width, int height, int radius);
void normalize_display(Mat src, string name);
Point2f intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2);
float dist(float x1, float y1, float x2, float y2);
void display_intersections(vector<Point2f> points, int width, int height);
Mat assign_gauss_lines(vector<Point2f> points, int width, int height);
Mat apply_thresh_int(Mat src, float thresh);
vector<Point2f> get_thresh_pos(Mat src);
void draw_features(vector<Point2f> points, Mat frame);
float get_radius(Point2f c);
Mat normalize(Mat src);
/** Global variables */

//---------Haar Globals-----------
String cascade_name = "dartcascade.xml";
CascadeClassifier cascade;

//----------Results from detectors---------
vector<Vec3f> circles;//Stores detected circles
vector<Rect> faces; //Stores detected objects
vector<Vec4i> lines; //Stores results


/// Global Variables
Mat hsv; Mat hue;
int bins = 25;

void Hist_and_Backproj(Mat src)
{
  MatND hist;
  int histSize = MAX( bins, 2 );
  float hue_range[] = { 0, 180 };
  const float* ranges = { hue_range };
    cvtColor( src, hsv, CV_BGR2HSV );

  /// Use only the Hue value
  hue.create( hsv.size(), hsv.depth() );
  int ch[] = { 0, 0 };
  mixChannels( &hsv, 1, &hue, 1, ch, 1 );
  /// Get the Histogram and normalize it
  calcHist( &hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false );
  normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

  /// Get Backprojection
  MatND backproj;
  calcBackProject( &hue, 1, 0, hist, backproj, &ranges, 1, true );

  /// Draw the backproj
  imshow( "BackProj", backproj );

  /// Draw the histogram
  int w = 400; int h = 400;
  int bin_w = cvRound( (double) w / histSize );
  Mat histImg = Mat::zeros( w, h, CV_8UC3 );

  for( int i = 0; i < bins; i ++ )
     { rectangle( histImg, Point( i*bin_w, h ), Point( (i+1)*bin_w, h - cvRound( hist.at<float>(i)*h/255.0 ) ), Scalar( 0, 0, 255 ), -1 ); }

  imshow( "Histogram", histImg );
}


int main( int argc, const char** argv ){ 
    char input;
    for(int i=0;i<16;i++){
    
    string s  = int_to_string(i); //current image index

    //Read source image
    Mat frame = imread((string)"dart"+s+(string)".jpg", CV_LOAD_IMAGE_COLOR);

    if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    
    //Show source
    //imshow((string)"dart",frame);
    
    //Apply detection and display results
    detectAndDisplay(frame);
    
    //Hold the windows open
    input=waitKey();
    //if you press x terminate all
    if(input=='x'){
        i=16;
    }
  }

  return 0;
};

//Apply all detections and display all
void detectAndDisplay(Mat frame){
    haarDetect(frame);
    vector<Point2f> faces_pos = convert_faces_to_points();
    Mat haar_gauss = assign_gauss(faces_pos, frame.cols, frame.rows, 125);
    normalize_display(haar_gauss, "haar_gauss");
    houghCircles(frame);
    vector<Point2f> circles_pos = convert_circles_to_points();
    Mat hough_circles_gauss = assign_gauss(circles_pos, frame.cols, frame.rows, 125);
    normalize_display(hough_circles_gauss, "hough_circles_gauss");
    houghLines(frame);
    vector<Point2f> lines_pos = compute_intersection_points();
    //display_intersections(lines_pos, frame.cols, frame.rows);
    Mat intersection_gauss = assign_gauss_lines(lines_pos, frame.cols, frame.rows);
    //normalize_display(intersection_gauss, "intersection_gauss");
    Mat thresholded_int = apply_thresh_int(intersection_gauss, 240);
    imshow("th",thresholded_int);
    vector<Point2f> tresh_pos = get_thresh_pos(thresholded_int);
    Mat gauss_thesh_pos = assign_gauss(tresh_pos, frame.cols, frame.rows, 125);
    //normalize_display(gauss_thesh_pos, "gaus_pos");
    Mat res1, res2;
    Mat haar_gauss_norm = normalize(haar_gauss);
    Mat hough_circles_gauss_norm = normalize(hough_circles_gauss);
    Mat gauss_thesh_pos_norm = normalize(gauss_thesh_pos);
    res1 = add_matrix(haar_gauss_norm, hough_circles_gauss_norm, 1, 1);
    res2 = add_matrix(res1, gauss_thesh_pos_norm, 1, 1);
    normalize_display(res2, "FINAL");
    Mat thresh_final = apply_thresh_int(res2, 240);
    imshow("TheshFinal", thresh_final);
    vector<Point2f> final_tresh_pos = get_thresh_pos(thresh_final);
    draw_features(final_tresh_pos, frame);

    //detectContours(frame);
    //surfDetect(frame);

};


//--------------------------------HELPERS-----------------------------
Mat normalize(Mat src){
    float minimum=10000, maximum=-10000;
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            if(src.at<float>(i,j)<minimum){
                minimum = src.at<float>(i,j);
            }
            if(src.at<float>(i,j)>maximum){
                maximum = src.at<float>(i,j);
            }
        }    
    }

    Mat m = Mat::zeros(src.rows, src.cols, CV_32F);
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            src.at<float>(i,j)-=minimum;
            m.at<float>(i,j) = ((src.at<float>(i,j)-minimum)/(maximum-minimum))*255;
        }
    }
    return m;
};


void draw_features(vector<Point2f> points, Mat frame){
    for(int i=0;i<points.size();i++){
        circle(frame, Point2f(points[i].y, points[i].x), get_radius(points[i]), Scalar(0,0,255), 3, 8, 0 );
        //cout<<get_radius(points[i])<<"\n";
    }
    for(int i=0;i<faces.size();i++){
    rectangle(frame, Point(faces[i].x, faces[i].y), 
            Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
    }
    imshow("DETECTED", frame);
};

float get_radius(Point2f c){
    float max_radius = -10000;
    for(int i=0;i<faces.size();i++){
        if(dist(faces[i].y+faces[i].height/2, faces[i].x+faces[i].width/2, c.x, c.y)<50){
            if(faces[i].width/2>max_radius)
                max_radius = faces[i].width/2;
            
        }
    }
    if(max_radius>0){
        return max_radius;
    }
    else{
        return 100;
    }
};


Mat add_matrix(Mat m1, Mat m2, float w1, float w2){
    Mat m3 = Mat::zeros(m1.rows, m1.cols, CV_32F);
    for(int i =0; i<m1.rows; i++){
        for(int j=0; j<m1.cols; j++){
            m3.at<float>(i,j) = m1.at<float>(i,j)*w1 + m2.at<float>(i,j)*w2;
        }
    }
    return m3;
}

//Convert int to string
string int_to_string(int i){
    std::ostringstream oss;
    oss << i;
    string s = oss.str();
    return s;
};

//Generate a gaussian distribution image
Mat genGauss(float sigma, int gaussSize){
    Mat kernelX = getGaussianKernel(gaussSize, sigma, CV_32F);
    Mat kernelY = getGaussianKernel(gaussSize, sigma, CV_32F);
    Mat kernelXY = kernelX*kernelY.t();
    return kernelXY;
};

//Compute center of rectangles and return contaning array
vector<Point2f> convert_faces_to_points(){
    vector<Point2f> r;
    int x,y;
    for(int i=0;i<faces.size();i++){
        y = faces[i].x + faces[i].width/2;
        x = faces[i].y + faces[i].height/2;
        r.push_back(Point2f(x, y));
    }
    return r;
};

vector<Point2f> convert_circles_to_points(){
    vector<Point2f> r;
    for(int i=0;i<circles.size();i++){
        r.push_back(Point2f(circles[i][1], circles[i][0]));
    }
    return r;
};

vector<Point2f> get_thresh_pos(Mat src){
    //Stores resulting data
     RNG rng(12345);//random number seed
    vector<Point2f> r;
    Point2f circleCentre;
    vector<vector<Point> > contours;
    vector<Vec4i> hier;
    vector<vector<Point> > contours_poly(contours.size());
    Mat canny_output;
    float radius;
    Canny(src, canny_output, 100, 100*2, 3);
    //Find contours
    findContours(canny_output, contours, hier, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    //  //Draw contours
    // Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );

    // for( int i = 0; i< contours.size(); i++ ){
    //     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) ); //This is a colour
    //     drawContours( drawing, contours, i, color, 2, 8, hier, 0, Point() );
    // }
    // //Display result   
    // imshow( "Contours", drawing );

    if(contours.size()>0){
        for (int i = 0; i < contours.size(); i++){
                //Find minimum circle
                //approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
                //Compute its radius and position
                minEnclosingCircle( (Mat)contours[i], circleCentre, radius);
                r.push_back(Point2f(circleCentre.y, circleCentre.x));
        }
    }
    return r;
};

float dist(float x1, float y1, float x2, float y2){
 return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}


Point2f intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2){
    Point2f r;
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;
    //Check if lines intersect
    float e  =  4.0f;
    float cross = d1.x*d2.y - d1.y*d2.x;
    //If lines parallel exit
    if (abs(cross) < /*EPS*/1e-8)
        return Point2f(-1,-1);

    //Compute intersection point
    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;

    //Check if intersection is on the segment itself
    float d11, d22, l;
    l=dist(o1.x, o1.y, p1.x, p1.y);
    d11 = dist(r.x, r.y, o1.x, o1.y);
    d22 = dist(r.x, r.y, p1.x, p1.y);
    if(d11+d22-e>l)
        return Point2f(-1,-1);
    //Return intersection
    return r;
}

vector<Point2f> compute_intersection_points(){
    Point2f p;
    vector<Point2f> iPoints;
    int  count = 0;
    for(int i = 0; i < lines.size(); i++){
        Vec4i l1 = lines[i];
        for(int j =0; j < lines.size(); j++){
            Vec4i l2 = lines[j];
            p = intersection(Point(l1[0],l1[1]),Point(l1[2],l1[3]),Point(l2[0],l2[1]),Point(l2[2],l2[3]));
            if (p.x!=-1){
                iPoints.push_back(Point2f(p.y, p.x));
            }
        }
    }
    return iPoints;
};

Mat assign_gauss(vector<Point2f> points, int width, int height, int size){
    Mat r = Mat::zeros(height, width, CV_32F);
    Mat g;
    int g_r = size;
    for(int i=0;i<points.size();i++){
        g = genGauss(6*g_r/25,g_r*2);
        int x = points[i].x;
        int y = points[i].y;
        for(int j = 0; j < g.rows; j++){
            for(int k = 0; k < g.cols; k++){
                if(x-g_r+j > 0 && 
                    y-g_r+k > 0 &&
                    x-g_r+j< height && 
                    y-g_r+k < width){
                        r.at<float>(x-g_r+j,y-g_r+k)+= 
                            g.at<float>(j,k);
                }
            }
        }
    }
    return r;
};

Mat assign_gauss_lines(vector<Point2f> points, int width, int height){
    Mat r = Mat::zeros(height, width, CV_32F);
    Mat g;
    int g_r = 25;
    for(int i=0;i<points.size();i++){
        g = genGauss(6,g_r*2);
        int x = points[i].x;
        int y = points[i].y;
        for(int j = 0; j < g.rows; j++){
            for(int k = 0; k < g.cols; k++){
                if(x-g_r+j > 0 && 
                    y-g_r+k > 0 &&
                    x-g_r+j< height && 
                    y-g_r+k < width){
                        r.at<float>(x-g_r+j,y-g_r+k)+= 
                            g.at<float>(j,k);
                }
            }
        }
    }
    return r;
};

void normalize_display(Mat src, string name){
    float minimum=10000, maximum=-10000;
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            if(src.at<float>(i,j)<minimum){
                minimum = src.at<float>(i,j);
            }
            if(src.at<float>(i,j)>maximum){
                maximum = src.at<float>(i,j);
            }
        }    
    }

    Mat m = Mat::zeros(src.rows, src.cols, CV_8U) ;
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            src.at<float>(i,j)-=minimum;
            m.at<uchar>(i,j) = ((src.at<float>(i,j)-minimum)/(maximum-minimum))*255;
        }
    }

    imshow(name, m);
};

void display_intersections(vector<Point2f> points, int width, int height){
    Mat m = Mat::zeros(height, width, CV_8U) ;
    for(int i=0;i<points.size();i++){
            if(points[i].x>0&&points[i].x<height&&points[i].y>0&&points[i].y<width)
            m.at<uchar>(points[i].x,points[i].y) = 255;
    }

    imshow("intersections_points", m);
};

Mat apply_thresh_int(Mat src, float thresh){
float minimum=10000, maximum=-10000;
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            if(src.at<float>(i,j)<minimum){
                minimum = src.at<float>(i,j);
            }
            if(src.at<float>(i,j)>maximum){
                maximum = src.at<float>(i,j);
            }
        }    
    }
    Mat m = Mat::zeros(src.rows, src.cols, CV_8U) ;
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            src.at<float>(i,j)-=minimum;
            m.at<uchar>(i,j) = ((src.at<float>(i,j)-minimum)/(maximum-minimum))*255;
            if(m.at<uchar>(i,j)>=thresh){
                m.at<uchar>(i,j)=255;
            }
            else{
                m.at<uchar>(i,j)=0;
            }
        }
    }
    return m;
};


//---------------------------DETECTORS---------------------------------
//Apply haar features detection
void haarDetect(Mat frame){
 
  //Apply preprocessing
  Mat frame_gray;
  cvtColor(frame, frame_gray, CV_BGR2GRAY);
  //imshow("Original Gray", frame_gray);
  equalizeHist(frame_gray, frame_gray);
  //imshow("Equalized Gray", frame_gray);

  //Apply cascade detection
  cascade.detectMultiScale( frame_gray, faces, 1.01, 2, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  //Draw the detected features
  Mat frame_tmp =frame.clone(); //make a copy of the original
  
  for( int i = 0; i < faces.size(); i++ ){
    //Draw rectangles
    rectangle(frame_tmp, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
  }

  //Display results
  //imshow((string)"dartH",frame_tmp);
};

//Detect circles using haar transform
void houghCircles(Mat frame){
 
  
  //Apply preprocessing
  Mat frame_gray;
  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );
  //imshow("equi", frame_gray);
  GaussianBlur( frame_gray, frame_gray, Size(9, 9), 2, 2 );

  // Apply the Hough Transform to find the circles
  HoughCircles( frame_gray, circles, CV_HOUGH_GRADIENT, 1, frame_gray.rows/2, 170, 20, 0, 0 );

  //Draw the detected features
  Mat frame_tmp =frame.clone(); //make a copy of the original
  
  for( int i = 0; i < circles.size(); i++ ){
    
    //Set-up variables
    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    int radius = cvRound(circles[i][2]);
    
    //Draw circles
    circle(frame_tmp, center, radius, Scalar(0,0,255), 3, 8, 0 );
  }

  //Display results
  //imshow((string)"dartCH",frame_tmp);
};

//Find lines
void houghLines(Mat frame){

  //Preprocessing
  Mat canny_output, frame_gray;
  Canny(frame, canny_output, 50, 150, 3);
  cvtColor(canny_output, frame_gray, CV_GRAY2BGR);

  //Apply transform
  HoughLinesP(canny_output, lines, 1, CV_PI/180, 50, 80, 10 );

  //Compute average line length
  float sum;
  float average;
  for(int i=0;i<lines.size();i++){
    sum+=sqrt((lines[i][0]-lines[i][2])*(lines[i][0]-lines[i][2])+(lines[i][1]-lines[i][3])*(lines[i][1]-lines[i][3]));
  }
  average = sum/lines.size();

  for( size_t i = 0; i < lines.size(); i++ ){
    Vec4i l = lines[i];
    line(frame_gray, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
  }
 
  //imshow("detected lines", frame_gray);
};

//Find all contours
void detectContours(Mat frame){
  RNG rng(12345);//random number seed

  //Stores resulting data
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  //Preprocessing
  //Detect edges using canny
  Mat canny_output;
  Canny(frame, canny_output, 100, 100*2, 3);
  //Find contours
  findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

  //Draw contours
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  vector<vector<Point> > contours_poly( contours.size() );
  for( int i = 0; i< contours.size(); i++ ){
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) ); //This is a colour
    drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
  }
  //Display result   
  imshow( "Contours", drawing );
  drawing = frame.clone();
  for( int i = 0; i< contours.size(); i++ ){
    approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
    if(contours_poly[i].size()==3){
      line(drawing, contours_poly[i][0], contours_poly[i][1], cvScalar(255,0,0),4);
      line(drawing, contours_poly[i][1], contours_poly[i][2], cvScalar(255,0,0),4);
      line(drawing, contours_poly[i][2], contours_poly[i][0], cvScalar(255,0,0),4);
    }
    if(contours_poly[i].size()==4){
      line(drawing, contours_poly[i][0], contours_poly[i][1], cvScalar(0,255,0),4);
      line(drawing, contours_poly[i][1], contours_poly[i][2], cvScalar(0,255,0),4);
      line(drawing, contours_poly[i][2], contours_poly[i][3], cvScalar(0,255,0),4);
      line(drawing, contours_poly[i][3], contours_poly[i][0], cvScalar(0,255,0),4);
    }
  }
  //Display result   
  imshow( "Contours2", drawing );
};

//Find SURF matches
void surfDetect(Mat img_scene){
  Mat img_object = imread( "dart.bmp", CV_LOAD_IMAGE_GRAYSCALE );

  //Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( img_object, keypoints_object );
  detector.detect( img_scene, keypoints_scene );

  //Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_object, descriptors_scene;

  extractor.compute( img_object, keypoints_object, descriptors_object );
  extractor.compute( img_scene, keypoints_scene, descriptors_scene );

  //Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 100;

  //Calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 3*min_dist )
     { good_matches.push_back( matches[i]); }
  }

  //-- Localize the object
  std::vector<Point2f> faces;
  std::vector<Point2f> scene;

  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    faces.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  //-- Draw keypoints ALL
  // Mat img_keypoints_2;
  //drawKeypoints( img_scene, keypoints_scene, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  // imshow("hpAll", img_keypoints_2);

  //Draw Matching Keypoints
  Mat frame_tmp = img_scene.clone();
  for( int i = 0; i < scene.size(); i++ ){
    
    //Set-up variables
    Point center(cvRound(scene[i].x), cvRound(scene[i].y));

    //Draw scene
    circle(frame_tmp, center, 5, Scalar(0,0,255), 3, 8, 0 );
  }

  //Display result
  imshow("SURF", frame_tmp);
};