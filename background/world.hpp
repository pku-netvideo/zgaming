#pragma once
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> // cv::Canny()
#define W 3840
#define H 2160
#define W_GT 1920
#define H_GT 1080
#define pi 3.1415926
using namespace cv;

class pixel {
public:
    /*
     r,g,b: the color of the pixel
     valid: indicating whether this pixel is covered by a color.
        valid == 0: not covered
        valid == 1: covered by image warping
        valid == 2: covered by inpainting algorithm
     */
    short r, g, b, valid;
    int human, timestamp;
    double depth;

    void clear()
    {
        r = g = b = valid = human = timestamp = depth = 0;
    }
};

struct viewpoint {
    /*
    x,y,z: the position of the viewpoint
    yaw , pitch , roll: the Eular angles of the viewpoint
    */
    double x, y, z, yaw, pitch, roll;
};

struct Location_3D
{
    double x, y, z;
};
