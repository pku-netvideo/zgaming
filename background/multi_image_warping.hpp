
//  perspective transform
//
//  Created by shanxigy on 8/23/20.
//  Copyright ?2020 shanxigy. All rights reserved.
//

#include <iostream>
#include <cstdio>
#include <math.h> 
#include "world.hpp"
#include "utils.h"
using namespace std;

//#define pi 3.1415926
//#define W 3840
//#define H 2160

class image_warping
{
private:
    int qx[W * H], qy[W * H], dist[W * H];
	bool changed_inpainting[H][W];

    /*
    Solve a linear equation system.
    */
    inline void gaussin(double(*a)[3], double* b, double* ans);

    /*
    * Rotate a vector by an Euler angles.
    * d: the destination vector
    * t0: the Eular angles
    * a: the source vector
    */
    inline void rotate(double d[3], double t0[3], double a[3]);

public:
    /*
    Nearest-neighbor inpainting algorithm.
    input:
    w: the width of the image
    h: the height of the image
    a: the input image
    max_distance: The pixels whose distance from the nearest valid pixel will not be filled.

    output:
    a: the processed image
    */
    inline void inpainting(int start_w, int end_w, int start_h, int end_h, pixel a[H][W], int max_distance = 10000);
	inline void inpainting(int change_cnt, int* change_h, int* change_w, pixel a[H][W], int corner[4][2]);
	inline void inpainting(int change_cnt, int* change_h, int* change_w, pixel a[H][W], int corner[4][2], int max_distance, Location_3D dst_location[H][W]);

    /*
    Given the source image with the viewpoint, output the image from a different viewpoint.

    input:
    w: The width of the image (in pixels)
    h: The height of the image (in pixels)
    start_w, end_w, start_h, end_h: The region of the source frame you want to warping.
    wFoV: The horizontal size of FoV (in degrees)
    hFoV: The vertical size of FoV (in degrees)
    src: The input image
    dst: The output image
    depth: Depth information of the input image, should be of the same size of src
    src_viewpoint: the position and the orientation of the source viewpoint.
    dst_viewpoint: the position and the orientation of the destination viewpoint.
    timestamp: represents the order of the input reference frames.
    dst_location: The 3D location of the pixels in the destination image (dst). Makes sense only in case of pixel.valid = 1.
    src_location: The 3D location of the pixels in the source image (src).
     output:
    dst: the output image, should be of the same size of src
    */
    inline void perspective_transform(
        int w, int h,
        int start_w, int end_w, int start_h, int end_h,
        double src_wFoV, double src_hFoV,
        double dst_wFoV, double dst_hFoV,
        pixel src[H][W], pixel dst[H][W],
        viewpoint src_viewpoint, viewpoint dst_viewpoint,
        double distance_to_plane, int timestamp,
        Location_3D src_location[H][W],
        Location_3D dst_location[H][W]
    );

    inline int perspective_transform_block(
        int src_w, int src_h,
        int dst_w, int dst_h,
        int start_w, int end_w, int start_h, int end_h,
        double src_wFoV, double src_hFoV,
        double dst_wFoV, double dst_hFoV,
        pixel src[H][W], pixel dst[H][W],
        viewpoint src_viewpoint, viewpoint dst_viewpoint,
        double distance_to_plane, int timestamp,
        Location_3D dst_location[H][W], bool changed[H][W], int change_h[W * H], int change_w[W * H],
        Mat MatGT, int mask[H_GT][W_GT], pixel dst_inpainting[H][W], double& psnr_block, double& psnr_inpaint
    );

    inline void get_YUV(pixel p, double& R, double& G, double& B, double& Y, double& U, double& V);

    inline void add_Y_channel(pixel& p, double bias);

    inline void RGB2YUV(double& R, double& G, double& B, double& Y, double& U, double& V);

    inline void YUV2RGB(double& R, double& G, double& B, double& Y, double& U, double& V);

    /*
    estimate the upsampling ratio.

    input:
    src_w: The width of the source image (in pixels)
    src_h: The height of the source  image (in pixels)
    dst_w: The width of the destination image (in pixels)
    dst_h: The height of the destination image (in pixels)
    start_w, end_w, start_h, end_h: The region of the source frame you want to warping.
    src_wFoV: The horizontal size of source FoV (in degrees)
    src_hFoV: The vertical size of source FoV (in degrees)
    src_wFoV: The horizontal size of destination FoV (in degrees)
    src_hFoV: The vertical size of destination FoV (in degrees)
    src: The input image
    src_viewpoint: the position and the orientation of the source viewpoint.
    dst_viewpoint: the position and the orientation of the destination viewpoint.

    output:
    the upsampling ratio.
    */
    inline double block_super_resolution_estimation(
        int src_w, int src_h,
        int dst_w, int dst_h,
        int start_w, int end_w, int start_h, int end_h,
        double src_wFoV, double src_hFoV,
        double dst_wFoV, double dst_hFoV,
        pixel src[H][W],
        viewpoint src_viewpoint, viewpoint dst_viewpoint,
        double distance_to_plane
    );

    /*
    estimate the upsampling ratio.

    input:
    src_w: The width of the source image (in pixels)
    src_h: The height of the source  image (in pixels)
    dst_w: The width of the destination image (in pixels)
    dst_h: The height of the destination image (in pixels)
    start_w, end_w, start_h, end_h: The region of the source frame you want to warping.
    src_wFoV: The horizontal size of source FoV (in degrees)
    src_hFoV: The vertical size of source FoV (in degrees)
    src_wFoV: The horizontal size of destination FoV (in degrees)
    src_hFoV: The vertical size of destination FoV (in degrees)
    src: The input image
    src_viewpoint: the position and the orientation of the source viewpoint.
    dst_viewpoint: the position and the orientation of the destination viewpoint.

    output:
    corner_2D: 8 integers corresponding to x1,x2,x3,x4,y1,y2,y3,y4
    */
    inline void block_corner_estimation(
        int src_w, int src_h,
        int dst_w, int dst_h,
        int start_w, int end_w, int start_h, int end_h,
        double src_wFoV, double src_hFoV,
        double dst_wFoV, double dst_hFoV,
        pixel src[H][W],
        viewpoint src_viewpoint, viewpoint dst_viewpoint,
        double distance_to_plane,
        int* corner_2D
    );
};
