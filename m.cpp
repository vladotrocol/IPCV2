//Program to compute image motion vectors

#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

using namespace cv;
using namespace std;


// 1. gesture detection using circle position and mean vector
// 2. tracking individual objects using closest neighbour 

Mat kx;
Mat ky;
Mat prevFrame;
bool init = false;
bool have_circle = false;
float thresh=110;
int siz = 20;
vector<vector<Point> > contours;
/// Initialize arguments for the filter
Point  anchor = Point( -1, -1 );
double  delta = 0;
int  ddepth = -1;


float fkX[3][3] = {
                   {1,0,-1},
                   {2,0,-2},
                   {1,0,-1}
                  };

//Values for y-derivative kernel
float fkY[3][3] = { 
                   {1,2,1},
                   {0,0,0},
                   {-1,-2,-1}
                  };
  
Mat derX(Mat src){
    Mat dst;
    filter2D(src, dst, ddepth , kx, anchor, delta, BORDER_DEFAULT );
    return dst;
};

Mat derY(Mat src){
    Mat dst;
    filter2D(src, dst, ddepth , ky, anchor, delta, BORDER_DEFAULT );
    return dst;
};

// void apply_temporal_thresh(Mat src){
//   Mat r = src.clone();
//   for(int i=0;i<src.rows;i++){
//     for(int j=0;j<src.cols;j++){
//       if(src.at<float>(i,j)>thresh){
//         r.at<float>(i,j) = 1;
//       }
//       else{
//         r.at<float>(i,j) = 0;
//       }
//     }
//   }
//   imshow("haha", r);

// };

 vector<Point2i> get_points(Mat src){
  vector<Point2i> r;
  for(int i=0+siz/2;i<src.rows-siz/2;i+=siz){
    for(int j=0+siz/2;j<src.cols-siz/2;j+=siz){
      // int sum=0;
      // for(int k=i-siz/2;k<i+siz/2;k++){
      //   for(int l=j-siz/2;l<j+siz/2;l++){
      //     sum+=src.at<float>(i,j);
      //   }
      // }
      //if(sum/400>thresh){
        r.push_back(Point2i(i,j));
      //}
    }
  }
  return r;
 };

 // Mat padded_mat(Mat src, int n){
 //    Mat m = Mat::zeros(src.rows+n, src.cols+n, CV_8UC1);
 //    for(int i=n/2;i<m.rows-n/2;i++){
 //      for(int j=n/2;j<m.cols-n/2;j++){
 //        m.at<float>(i,j) = src.at<float>(i-n/2, j-n/2);
 //      }
 //    }
 //    return m;
 // };

 Mat compute_region(Point2i p, int n, Mat dx, Mat dy, Mat dt){
    float sum_dx2 = 0;
    float sum_dy2 = 0;
    float sum_dx_dy = 0;
    float sum_dx_dt = 0;
    float sum_dy_dt = 0;
    for(int i=p.x-n/2;i<p.x+n/2;i++){
      for(int j=p.y-n/2;j<p.y+n/2;j++){
        sum_dx2+=dx.at<float>(i,j)*dx.at<float>(i,j);
       // cout<<dx.at<float>(i,j)<<" "<<dx.at<float>(i,j)*dx.at<float>(i,j)<<" "<<sum_dx2<<"\n";
        sum_dy2+=dy.at<float>(i,j)*dy.at<float>(i,j);
        sum_dx_dy+=dx.at<float>(i,j)*dy.at<float>(i,j);
        sum_dx_dt+=dx.at<float>(i,j)*dt.at<float>(i,j);
        sum_dy_dt+=dy.at<float>(i,j)*dt.at<float>(i,j); 
      }
    }
    Mat A = Mat::zeros(2, 2, CV_32F);
    A.at<float>(0,0) = sum_dx2;
    A.at<float>(0,1) = sum_dx_dy;
    A.at<float>(1,0) = sum_dx_dy;
    A.at<float>(1,1) = sum_dy2;
    Mat Ap = A.inv(DECOMP_LU);
    Mat b = Mat::zeros(2, 1, CV_32F);
    b.at<float>(0,0) = -sum_dx_dt;
    b.at<float>(1,0) = -sum_dy_dt;
    //cout<<"a:"<<A<<"\n\n"<<"b:"<<b<<"\n\n";
    return Ap*b;
 };





void get_thresh_pos(Mat src, Mat f){
    //Stores resulting data
     RNG rng(12345);//random number seed
    vector<Point2f> r;
    Point2f circleCentre;
    
    vector<Vec4i> hier;
    vector<vector<Point> > contours_poly(contours.size());
    Mat canny_output;
    float radius;
    //Canny(src, canny_output, 100, 100*2, 3);
    //Find contours
    
    Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(40, 40), Point(2,2));
    Mat resultDilate;
    dilate( src, resultDilate, element );
    imshow("circlwe", resultDilate);
    findContours(resultDilate, contours, hier, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
     //Draw contours
    Mat drawing = Mat::zeros( src.size(), CV_8UC3 );

    for( int i = 0; i< contours.size(); i++ ){
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) ); //This is a colour
        drawContours( drawing, contours, i, color, 2, 8, hier, 0, Point() );
    }
 
    imshow( "Contourss", drawing );

    if(contours.size()>0){
	//char buffer [10];
	 
        for (int i = 0; i < contours.size(); i++){
		std::stringstream ss;
		ss << i;
            //Rect R = boundingRect( Mat(contours_poly[i]) );
            //rectangle( src, R.tl(), R.br(), Scalar(0,0,255), 2, 8, 0 );
                //Find minimum circle
                //approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
                //Compute its radius and position
                minEnclosingCircle( (Mat)contours[i], circleCentre, radius);
                if(radius>110){ // th
		have_circle = true;
                circle(f, circleCentre, radius, Scalar(0,0,255), 1, 8, 0 );
		//itoa(i,buffer,10);
		putText(f, ss.str(), circleCentre, FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
		}
		else
		have_circle = false;
                //r.push_back(Point2f(circleCentre.y, circleCentre.x));
        }
    }
    imshow("circle", f);
};




int main(int, char**)
{
    kx = Mat(3, 3, CV_32FC1, &fkX);
    ky = Mat(3, 3, CV_32FC1, &fkY);

      VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    //namedWindow("Camera",1);
    for(;;) //while(1)
    {   
         Mat frame;
         Mat frame_gray;
         Mat frame_gray2;

         //Frame video
         cap >> frame;


         Mat f = frame.clone();
         cvtColor(frame, frame_gray2, CV_BGR2GRAY); 
          frame_gray2.convertTo(frame_gray, CV_32FC1);
          Mat acc = Mat::zeros(frame.rows, frame.cols, CV_8UC1);


        // cvConvertScale(&frame_gray2, &frame_gray, 1.0 / 255.0, 0.0);
        if(init){
            Mat derx, dery;   
            // get a new frame from camera

            //------------Important-----------
            derx = derX(frame_gray); //x spatial derivative Ix
            dery = derY(frame_gray); // y spatial derivative Iy
            Mat tilda = prevFrame - frame_gray; // temporal derivative It
            
            //-----------Flexibility.. vector of points instead of grid----
            vector<Point2i> p = get_points(tilda);

            vector<Point2f> v;
            Mat reg;
            float vx, vy;
            int px, py;

            for(vector<Point2i>::const_iterator i = p.begin(); i != p.end(); ++i) {
              
              //Get the movement vector for the current point/region
              reg = compute_region(*i, siz, derx, dery, tilda);
              

              //Compute the new size of the vector
              vx = reg.at<float>(0,0);
              vy = reg.at<float>(1,0);
              float mag =  sqrt(vx*vx + vy*vy);
              vx /= mag;
              vy /= mag;
              px = (int)((float)i->y + vx * (mag + siz/2)); //new sizes
              py = (int)((float)i->x + vy * (mag + siz/2)); //new sizes

              //mat to point vel vector
              v.push_back(Point2f(reg.at<float>(0,0), reg.at<float>(1,0)));


              if(mag>0.5f){
                //Draw vector main line
                line(frame, Point(i->y, i->x), Point(px,py), Scalar(0,0,255), 1, CV_AA);
                
                //Draw vector arrow
                double angle = atan2((double)i->x-py, (double)i->y-px);
                int ax = (int) (px + 4* cos(angle + 3.1415/4));
                int ay = (int) (py + 4* sin(angle + 3.1415/4));
                int bx = (int) (px + 4* cos(angle - 3.1415/4));
                int by = (int) (py + 4* cos(angle - 3.1415/4));
                line(frame, Point(ax,ay), Point(px,py), Scalar(0,0,255), 1, CV_AA);
                line(frame, Point(bx,by), Point(px,py), Scalar(0,0,255), 1, CV_AA);
                
      
                for(int k=i->x-siz/2;k<i->x+siz/2;k++){
                  for(int l=i->y-siz/2;l<i->y+siz/2;l++){
                      acc.at<uchar>(k, l) = 255;                
                  }
                }

              }
              else{
                 circle(frame, Point(i->y,i->x), 1, Scalar(0,0,255), 1, 8, 0 );
              }
            }

            //Compute mean vector
            float mx=0, my=0;
            for(vector<Point2f>::const_iterator i = v.begin(); i != v.end(); ++i) {
              mx+=i->x;
              my+=i->y;
            }
            mx/=v.size();
            my/=v.size();

	    //cout << mx << " " << my << "\n";

	    if (mx < 0 && have_circle == true && mx > -0.3)
		putText(frame, "RIGHT", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
	    if (mx > 0 && have_circle == true && mx < 0.3)
		putText(frame, "LEFT", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
            //draw mean vector
            line(frame, Point(frame.cols/2, frame.rows/2), Point(frame.cols/2+mx*frame.cols, frame.rows/2+my*frame.rows), Scalar(0,255,0), 8, CV_AA);


            //
            Mat a, b, c, d;
            frame_gray.convertTo(a, CV_8UC1);
            derx.convertTo(b, CV_8UC1);
            dery.convertTo(c, CV_8UC1);
            tilda.convertTo(d, CV_8UC1);

            //imshow("acc", acc);
            //imshow("asda", frame);

            //Get position of detected circle
            get_thresh_pos(acc, frame);
            
             imshow("Camera", a);
             imshow("derx", b);
            imshow("dery", c);
            imshow("temporal", d);
            prevFrame = frame_gray;
            if(waitKey(30) >= 0) break;
        }
        else{
            prevFrame = frame_gray;
            init = true;
        }
    }

    return 0;
}
