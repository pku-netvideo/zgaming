//
//  perspective transform
//
//  Created by shanxigy on 8/23/20.
//  Copyright ?2020 shanxigy. All rights reserved.
//

#include <iostream>
#include <cstdio>
#include <minmax.h> 
//#include "world.hpp"
#include "multi_image_warping.hpp"
using namespace std;

void image_warping::gaussin(double(*a)[3], double* b, double* ans)
{
    int n = 3;

    int i, j, k;
    double c[3];
    for (k = 0; k < n - 1; k++)
    {
        for (i = k + 1; i < n; i++)
            c[i] = a[i][k] / a[k][k];

        for (i = k + 1; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                a[i][j] = a[i][j] - c[i] * a[k][j];
            }
            b[i] = b[i] - c[i] * b[k];
        }
    }

    double x[3];
    x[n - 1] = b[n - 1] / a[n - 1][n - 1];
    for (i = n - 2; i >= 0; i--)
    {
        double sum = 0;
        for (j = i + 1; j < n; j++)
        {
            sum += a[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / a[i][i];
    }

    for (i = 0; i < n; i++)
        ans[i] = x[i];
}

inline void image_warping::rotate(double d[3], double t0[3], double a[3]) {
    double t[3];
    for (int i = 0; i < 3; i++)
        t[i] = t0[i] * pi / 180.0;
    d[0] = d[1] = d[2] = 0;
    d[0] = cos(t[2]) * (cos(t[1]) * a[0] + sin(t[1]) * (sin(t[0]) * a[1] + cos(t[0]) * a[2])) - (sin(t[2]) * (cos(t[0]) * a[1] - sin(t[0]) * a[2]));
    d[1] = sin(t[2]) * (cos(t[1]) * a[0] + sin(t[1]) * (sin(t[0]) * a[1] + cos(t[0]) * a[2])) + (cos(t[2]) * (cos(t[0]) * a[1] - sin(t[0]) * a[2]));
    d[2] = -sin(t[1]) * a[0] + cos(t[1]) * (sin(t[0]) * a[1] + cos(t[0]) * a[2]);
}

inline void image_warping::inpainting(int start_w, int end_w, int start_h, int end_h, pixel a[H][W], int max_distance)
{
    int l, r, px, py;
    int c[4][2] = { {1,0},{-1,0},{0,1},{0,-1} };
    l = r = 0;
    for (int i = start_h; i < end_h; ++i)
        for (int j = start_w; j < end_w; ++j)
            if (a[i][j].valid == 1) {
                qx[r] = i;
                qy[r] = j;
                dist[r++] = 0;
            }
            else a[i][j].valid = 0;
    while (l < r) {
        if (dist[l] >= max_distance) break;
        for (int i = 0; i < 4; ++i) {
            px = qx[l] + c[i][0];
            py = qy[l] + c[i][1];
            if (px >= start_h && px < end_h && py >= start_w && py < end_w && !a[px][py].valid) {
                a[px][py].r = a[qx[l]][qy[l]].r;
                a[px][py].g = a[qx[l]][qy[l]].g;
                a[px][py].b = a[qx[l]][qy[l]].b;
				a[px][py].depth = a[qx[l]][qy[l]].depth;
                a[px][py].valid = 2;
                qx[r] = px;
                qy[r] = py;
                dist[r++] = dist[l] + 1;
            }
        }
        ++l;
    }
}

inline void image_warping::inpainting(int change_cnt, int* change_h, int* change_w, pixel a[H][W], int corner[4][2], int max_distance, Location_3D dst_location[H][W])
{
	int l, r, px, py;
	int c[4][2] = { { 1,0 },{ -1,0 },{ 0,1 },{ 0,-1 } };
	int edge_p[2];
	/*for (int i = 0; i < 4; i++) {
		corner[i][0] = min(H, max(0, corner[i][0]));
		corner[i][1] = min(W, max(0, corner[i][1]));
	}*/
	/*memset(changed_inpainting, false, sizeof(changed_inpainting));*/
	l = r = 0;
	for (int i = 0; i < change_cnt; ++i) {
		qx[r] = change_h[i];
		qy[r] = change_w[i];
		/*changed_inpainting[change_h[i]][change_w[i]] = true;*/
		dist[r++] = 0;
	}
	while (l < r) {
		if (dist[l] >= max_distance) break;
		for (int i = 0; i < 4; ++i) {
			px = qx[l] + c[i][0];
			py = qy[l] + c[i][1];
			edge_p[0] = px;
			edge_p[1] = py;
			if (px >= 0 && px < H && py >= 0 && py < W && IsPointInsideShape(edge_p, corner) && (a[px][py].valid == 0 || sqrt(a[px][py].depth) - sqrt(a[qx[l]][qy[l]].depth) > 0.3)/* && a[px][py].valid != 1*//* && !changed_inpainting[px][py]*/) {
				a[px][py].r = a[qx[l]][qy[l]].r;
				a[px][py].g = a[qx[l]][qy[l]].g;
				a[px][py].b = a[qx[l]][qy[l]].b;
				a[px][py].depth = a[qx[l]][qy[l]].depth;
				a[px][py].human = a[qx[l]][qy[l]].human;
				a[px][py].timestamp = a[qx[l]][qy[l]].timestamp;
				dst_location[px][py].x = dst_location[qx[l]][qy[l]].x;
				dst_location[px][py].y = dst_location[qx[l]][qy[l]].x;
				dst_location[px][py].z = dst_location[qx[l]][qy[l]].x;
				if (a[px][py].valid == 0)
					a[px][py].valid = 2;
				else if (a[px][py].valid == 1)
					a[px][py].valid = 1;
				else if (a[px][py].valid == 2)
					a[px][py].valid = 1;
				/*changed_inpainting[px][py] = true;*/
				qx[r] = px;
				qy[r] = py;
				dist[r++] = dist[l] + 1;
			}
		}
		++l;
	}
}

inline void image_warping::inpainting(int change_cnt, int* change_h, int* change_w, pixel a[H][W], int corner[4][2])
{
	int l, r, px, py;
	int c[4][2] = { { 1,0 },{ -1,0 },{ 0,1 },{ 0,-1 } };
	int edge_p[2];
	/*memset(changed_inpainting, false, sizeof(changed_inpainting));*/
	l = r = 0;
	for (int i = 0; i < change_cnt; ++i) {
		qx[r] = change_h[i];
		qy[r] = change_w[i];
		/*changed_inpainting[change_h[i]][change_w[i]] = true;*/
		r++;
	}
	while (l < r) {
		for (int i = 0; i < 4; ++i) {
			px = qx[l] + c[i][0];
			py = qy[l] + c[i][1];
			edge_p[0] = px;
			edge_p[1] = py;
			/*if (IsPointInsideShape(edge_p, corner))
				cout << "in" << endl;
			else
				cout << "out" << endl;*/
			if (px >= 0 && px < H && py >= 0 && py < W && IsPointInsideShape(edge_p, corner) && (a[px][py].valid == 0 || a[px][py].depth - a[qx[l]][qy[l]].depth > 0.05) && a[px][py].valid != 1/* && !changed_inpainting[px][py]*/) {
				a[px][py].r = a[qx[l]][qy[l]].r;
				a[px][py].g = a[qx[l]][qy[l]].g;
				a[px][py].b = a[qx[l]][qy[l]].b;
				a[px][py].depth = a[qx[l]][qy[l]].depth;
				a[px][py].valid = 2;
				/*changed_inpainting[px][py] = true;*/
				qx[r] = px;
				qy[r] = py;
				r++;
			}
		}
		++l;
	}
}

void image_warping::perspective_transform(
    int w, int h,
    int start_w, int end_w, int start_h, int end_h,
    double src_wFoV, double src_hFoV,
    double dst_wFoV, double dst_hFoV,
    pixel src[H][W], pixel dst[H][W],
    viewpoint src_viewpoint, viewpoint dst_viewpoint,
    double distance_to_plane, int timestamp,
    Location_3D src_location[H][W],
    Location_3D dst_location[H][W]
)
{
    //These are temperal variations used for vector rotation (by calling the function rotate()).
    double dst_vector[3], src_vector[3], Euler[3];

    //Euler angles of the source viewpoint.
    Euler[0] = src_viewpoint.pitch;
    Euler[1] = src_viewpoint.roll;
    Euler[2] = src_viewpoint.yaw;

    //Computing the direction vector of the source viewpoint.
    src_vector[0] = 0, src_vector[1] = 1, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double src_viewpoint_dx = dst_vector[0];
    double src_viewpoint_dy = dst_vector[1];
    double src_viewpoint_dz = dst_vector[2];

    //Computing the horizontal vector of the source viewpoint.
    src_vector[0] = 1, src_vector[1] = 0, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double horizontal_src_x = dst_vector[0];
    double horizontal_src_y = dst_vector[1];
    double horizontal_src_z = dst_vector[2];

    //Computing the vertical vector of the source viewpoint.
    src_vector[0] = 0, src_vector[1] = 0, src_vector[2] = -1;
    rotate(dst_vector, Euler, src_vector);
    double vertical_src_x = dst_vector[0];
    double vertical_src_y = dst_vector[1];
    double vertical_src_z = dst_vector[2];

    //Euler angles of the destination viewpoint.
    Euler[0] = dst_viewpoint.pitch;
    Euler[1] = dst_viewpoint.roll;
    Euler[2] = dst_viewpoint.yaw;

    //Computing the direction vector of the destination viewpoint.
    src_vector[0] = 0, src_vector[1] = 1, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double dst_viewpoint_dx = dst_vector[0];
    double dst_viewpoint_dy = dst_vector[1];
    double dst_viewpoint_dz = dst_vector[2];

    //Computing the horizontal vector of the destination viewpoint.
    src_vector[0] = 1, src_vector[1] = 0, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double horizontal_dst_x = dst_vector[0];
    double horizontal_dst_y = dst_vector[1];
    double horizontal_dst_z = dst_vector[2];

    //Computing the vertical vector of the destination viewpoint.
    src_vector[0] = 0, src_vector[1] = 0, src_vector[2] = -1;
    rotate(dst_vector, Euler, src_vector);
    double vertical_dst_x = dst_vector[0];
    double vertical_dst_y = dst_vector[1];
    double vertical_dst_z = dst_vector[2];

    //normalization the vectors based on the value of distance_to_plane.
    src_viewpoint_dx *= distance_to_plane;
    src_viewpoint_dy *= distance_to_plane;
    src_viewpoint_dz *= distance_to_plane;
    dst_viewpoint_dx *= distance_to_plane;
    dst_viewpoint_dy *= distance_to_plane;
    dst_viewpoint_dz *= distance_to_plane;

    //computing the size of the source frame in 3D space.
    double src_distance_vertical = tan(src_hFoV / 180.0 * pi / 2) * distance_to_plane;
    double src_distance_horizontal = tan(src_wFoV / 180.0 * pi / 2) * distance_to_plane;
    double dst_distance_vertical = tan(dst_hFoV / 180.0 * pi / 2) * distance_to_plane;
    double dst_distance_horizontal = tan(dst_wFoV / 180.0 * pi / 2) * distance_to_plane;

    horizontal_src_x *= src_distance_horizontal;
    horizontal_src_y *= src_distance_horizontal;
    horizontal_src_z *= src_distance_horizontal;

    horizontal_dst_x *= dst_distance_horizontal;
    horizontal_dst_y *= dst_distance_horizontal;
    horizontal_dst_z *= dst_distance_horizontal;

    vertical_src_x *= src_distance_vertical;
    vertical_src_y *= src_distance_vertical;
    vertical_src_z *= src_distance_vertical;

    vertical_dst_x *= dst_distance_vertical;
    vertical_dst_y *= dst_distance_vertical;
    vertical_dst_z *= dst_distance_vertical;

    double src_horizontal_perpixel_x = horizontal_src_x / (w / 2.0);
    double src_horizontal_perpixel_y = horizontal_src_y / (w / 2.0);
    double src_horizontal_perpixel_z = horizontal_src_z / (w / 2.0);
    double src_vertical_perpixel_x = vertical_src_x / (h / 2.0);
    double src_vertical_perpixel_y = vertical_src_y / (h / 2.0);
    double src_vertical_perpixel_z = vertical_src_z / (h / 2.0);

    //Computing the per-pixel displacement in 3D space from the central point of the frame (both horizontal and vertical).
    double dst_horizontal_perpixel_x = horizontal_dst_x / (w / 2.0);
    double dst_horizontal_perpixel_y = horizontal_dst_y / (w / 2.0);
    double dst_horizontal_perpixel_z = horizontal_dst_z / (w / 2.0);
    double dst_vertical_perpixel_x = vertical_dst_x / (h / 2.0);
    double dst_vertical_perpixel_y = vertical_dst_y / (h / 2.0);
    double dst_vertical_perpixel_z = vertical_dst_z / (h / 2.0);

    //The size of the per-pixel displacement.
    double dst_horizontal_perpixel_size = sqrt((dst_horizontal_perpixel_x * dst_horizontal_perpixel_x) + (dst_horizontal_perpixel_y * dst_horizontal_perpixel_y) + (dst_horizontal_perpixel_z * dst_horizontal_perpixel_z));
    double dst_vertical_perpixel_size = sqrt((dst_vertical_perpixel_x * dst_vertical_perpixel_x) + (dst_vertical_perpixel_y * dst_vertical_perpixel_y) + (dst_vertical_perpixel_z * dst_vertical_perpixel_z));

    pixel point;
    int midw = w / 2;
    int midh = h / 2;
    double pos_x, pos_y, pos_z, delta_x, delta_y, delta_z, dst_x, dst_y, dst_z, distance, depth3D;
    double a[3][3], b[3], c[3], size, vx, vy, vz;
    int i1, j1;

    for (int i = start_h; i < end_h; ++i)
        for (int j = start_w; j < end_w; ++j) {
            //computing the coordinate of 3D point for each pixel in the original frame.
            delta_x = src_viewpoint_dx + (0.5 + j - midw) * src_horizontal_perpixel_x + (0.5 + i - midh) * src_vertical_perpixel_x;
            delta_y = src_viewpoint_dy + (0.5 + j - midw) * src_horizontal_perpixel_y + (0.5 + i - midh) * src_vertical_perpixel_y;
            delta_z = src_viewpoint_dz + (0.5 + j - midw) * src_horizontal_perpixel_z + (0.5 + i - midh) * src_vertical_perpixel_z;

            size = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
            depth3D = src[i][j].depth * size / distance_to_plane;

            //delta is the displacement from the source viewpoint to the 3D position of the pixel (i, j).
            delta_x = delta_x / size * depth3D;
            delta_y = delta_y / size * depth3D;
            delta_z = delta_z / size * depth3D;
            //pos is the 3D position of pixel (i, j).
            pos_x = src_viewpoint.x + delta_x;
            pos_y = src_viewpoint.y + delta_y;
            pos_z = src_viewpoint.z + delta_z;
            src_location[i][j].x = pos_x;
            src_location[i][j].y = pos_y;
            src_location[i][j].z = pos_z;

            //matrix a is the arguments of Linear simultaneous equations in three unknowns
            a[0][0] = pos_y - dst_viewpoint.y;
            a[0][1] = dst_viewpoint.x - pos_x;
            a[0][2] = 0;
            b[0] = -((dst_viewpoint.y - pos_y) * dst_viewpoint.x + (pos_x - dst_viewpoint.x) * dst_viewpoint.y);

            a[1][0] = pos_z - dst_viewpoint.z;
            a[1][1] = 0;
            a[1][2] = dst_viewpoint.x - pos_x;
            b[1] = -((dst_viewpoint.z - pos_z) * dst_viewpoint.x + (pos_x - dst_viewpoint.x) * dst_viewpoint.z);

            a[2][0] = dst_viewpoint_dx;
            a[2][1] = dst_viewpoint_dy;
            a[2][2] = dst_viewpoint_dz;
            b[2] = -(-(dst_viewpoint.x + dst_viewpoint_dx) * dst_viewpoint_dx
                - (dst_viewpoint.y + dst_viewpoint_dy) * dst_viewpoint_dy
                - (dst_viewpoint.z + dst_viewpoint_dz) * dst_viewpoint_dz);

            gaussin(a, b, c);
            dst_x = c[0], dst_y = c[1], dst_z = c[2];
            distance = (pos_x - dst_viewpoint.x) * (pos_x - dst_viewpoint.x) + (pos_y - dst_viewpoint.y) * (pos_y - dst_viewpoint.y) + (pos_z - dst_viewpoint.z) * (pos_z - dst_viewpoint.z);
            if ((dst_x - dst_viewpoint.x) * (pos_x - dst_viewpoint.x) + (dst_y - dst_viewpoint.y) * (pos_y - dst_viewpoint.y) + (dst_z - dst_viewpoint.z) * (pos_z - dst_viewpoint.z) < 0) continue;

            //(vx, vy, vz) is the displacement of the point (dst_x, dst_y, dst_z) from the central point of the dstination frame.
            vx = dst_x - (dst_viewpoint.x + dst_viewpoint_dx);
            vy = dst_y - (dst_viewpoint.y + dst_viewpoint_dy);
            vz = dst_z - (dst_viewpoint.z + dst_viewpoint_dz);

            //(i1, j1) is the position in the destination frame of the pixel (i, j) in the source frame.
            i1 = midh + round(-0.5 + (vx * vertical_dst_x + vy * vertical_dst_y + vz * vertical_dst_z) / dst_distance_vertical / dst_vertical_perpixel_size);
            j1 = midw + round(-0.5 + (vx * horizontal_dst_x + vy * horizontal_dst_y + vz * horizontal_dst_z) / dst_distance_horizontal / dst_horizontal_perpixel_size);

            //Copy the pixel value of (i, j) in the source frame to (i1, j1) in the destination frame.
            if (j1 >= 0 && j1 < w && i1 >= 0 && i1 < h) {
                if (dst[i1][j1].valid != 1 || timestamp > dst[i1][j1].timestamp || (timestamp == dst[i1][j1].timestamp && distance < dst[i1][j1].depth)) {
                    dst[i1][j1].r = src[i][j].r;
                    dst[i1][j1].g = src[i][j].g;
                    dst[i1][j1].b = src[i][j].b;
                    dst[i1][j1].valid = 1;
                    dst[i1][j1].human |= src[i][j].human;
                    dst[i1][j1].depth = distance;
                    dst[i1][j1].timestamp = timestamp;
                    dst_location[i1][j1].x = pos_x;
                    dst_location[i1][j1].y = pos_y;
                    dst_location[i1][j1].z = pos_z;
                }
            }
        }
}

void image_warping::RGB2YUV(double& R, double& G, double& B, double& Y, double& U, double& V) {
    Y = 0.299 * R + 0.587 * G + 0.114 * B;
    U = (B - Y) / 1.772;
    V = (R - Y) / 1.402;
}

void image_warping::YUV2RGB(double& R, double& G, double& B, double& Y, double& U, double& V) {
    R = Y + 1.4075 * V;
    G = Y - 0.3455 * U - 0.7169 * V;
    B = Y + 1.779 * U;
}

void image_warping::get_YUV(pixel p, double& R, double& G, double& B, double& Y, double& U, double& V)
{
    R = p.r;
    G = p.g;
    B = p.b;
    RGB2YUV(R, G, B, Y, U, V);
}

void image_warping::add_Y_channel(pixel& p, double bias)
{
    double R, G, B, Y, U, V;
    get_YUV(p, R, G, B, Y, U, V);
    Y += bias;
    YUV2RGB(R, G, B, Y, U, V);
    if (R >= 0 && R <= 255 && G >= 0 && G <= 255 && B >= 0 && B <= 255) {
            p.b = round(B);
            p.g = round(G);
            p.r = round(R);
    }
}

int image_warping::perspective_transform_block(
    int src_w, int src_h,
    int dst_w, int dst_h,
    int start_w, int end_w, int start_h, int end_h,
    double src_wFoV, double src_hFoV,
    double dst_wFoV, double dst_hFoV,
    pixel src[H][W], pixel dst[H][W],
    viewpoint src_viewpoint, viewpoint dst_viewpoint,
    double distance_to_plane, int timestamp,
    Location_3D dst_location[H][W], bool changed[H][W], int change_h[W*H], int change_w[W*H],
    Mat MatGT, int mask[H_GT][W_GT], pixel dst_inpainting[H][W], double& psnr_block, double& psnr_inpaint
)
{
    //These are temperal variations used for vector rotation (by calling the function rotate()).
    double dst_vector[3], src_vector[3], Euler[3];

    //Euler angles of the source viewpoint.
    Euler[0] = src_viewpoint.pitch;
    Euler[1] = src_viewpoint.roll;
    Euler[2] = src_viewpoint.yaw;

    //Computing the direction vector of the source viewpoint.
    src_vector[0] = 0, src_vector[1] = 1, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double src_viewpoint_dx = dst_vector[0];
    double src_viewpoint_dy = dst_vector[1];
    double src_viewpoint_dz = dst_vector[2];

    //Computing the horizontal vector of the source viewpoint.
    src_vector[0] = 1, src_vector[1] = 0, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double horizontal_src_x = dst_vector[0];
    double horizontal_src_y = dst_vector[1];
    double horizontal_src_z = dst_vector[2];

    //Computing the vertical vector of the source viewpoint.
    src_vector[0] = 0, src_vector[1] = 0, src_vector[2] = -1;
    rotate(dst_vector, Euler, src_vector);
    double vertical_src_x = dst_vector[0];
    double vertical_src_y = dst_vector[1];
    double vertical_src_z = dst_vector[2];

    //Euler angles of the destination viewpoint.
    Euler[0] = dst_viewpoint.pitch;
    Euler[1] = dst_viewpoint.roll;
    Euler[2] = dst_viewpoint.yaw;

    //Computing the direction vector of the destination viewpoint.
    src_vector[0] = 0, src_vector[1] = 1, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double dst_viewpoint_dx = dst_vector[0];
    double dst_viewpoint_dy = dst_vector[1];
    double dst_viewpoint_dz = dst_vector[2];

    //Computing the horizontal vector of the destination viewpoint.
    src_vector[0] = 1, src_vector[1] = 0, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double horizontal_dst_x = dst_vector[0];
    double horizontal_dst_y = dst_vector[1];
    double horizontal_dst_z = dst_vector[2];

    //Computing the vertical vector of the destination viewpoint.
    src_vector[0] = 0, src_vector[1] = 0, src_vector[2] = -1;
    rotate(dst_vector, Euler, src_vector);
    double vertical_dst_x = dst_vector[0];
    double vertical_dst_y = dst_vector[1];
    double vertical_dst_z = dst_vector[2];

    //normalization the vectors based on the value of distance_to_plane.
    src_viewpoint_dx *= distance_to_plane;
    src_viewpoint_dy *= distance_to_plane;
    src_viewpoint_dz *= distance_to_plane;
    dst_viewpoint_dx *= distance_to_plane;
    dst_viewpoint_dy *= distance_to_plane;
    dst_viewpoint_dz *= distance_to_plane;

    //computing the size of the source frame in 3D space.
    double src_distance_vertical = tan(src_hFoV / 180.0 * pi / 2) * distance_to_plane;
    double src_distance_horizontal = tan(src_wFoV / 180.0 * pi / 2) * distance_to_plane;
    double dst_distance_vertical = tan(dst_hFoV / 180.0 * pi / 2) * distance_to_plane;
    double dst_distance_horizontal = tan(dst_wFoV / 180.0 * pi / 2) * distance_to_plane;

    horizontal_src_x *= src_distance_horizontal;
    horizontal_src_y *= src_distance_horizontal;
    horizontal_src_z *= src_distance_horizontal;

    horizontal_dst_x *= dst_distance_horizontal;
    horizontal_dst_y *= dst_distance_horizontal;
    horizontal_dst_z *= dst_distance_horizontal;

    vertical_src_x *= src_distance_vertical;
    vertical_src_y *= src_distance_vertical;
    vertical_src_z *= src_distance_vertical;

    vertical_dst_x *= dst_distance_vertical;
    vertical_dst_y *= dst_distance_vertical;
    vertical_dst_z *= dst_distance_vertical;

    double src_horizontal_perpixel_x = horizontal_src_x / (src_w / 2.0);
    double src_horizontal_perpixel_y = horizontal_src_y / (src_w / 2.0);
    double src_horizontal_perpixel_z = horizontal_src_z / (src_w / 2.0);
    double src_vertical_perpixel_x = vertical_src_x / (src_h / 2.0);
    double src_vertical_perpixel_y = vertical_src_y / (src_h / 2.0);
    double src_vertical_perpixel_z = vertical_src_z / (src_h / 2.0);

    //Computing the per-pixel displacement in 3D space from the central point of the frame (both horizontal and vertical).
    double dst_horizontal_perpixel_x = horizontal_dst_x / (dst_w / 2.0);
    double dst_horizontal_perpixel_y = horizontal_dst_y / (dst_w / 2.0);
    double dst_horizontal_perpixel_z = horizontal_dst_z / (dst_w / 2.0);
    double dst_vertical_perpixel_x = vertical_dst_x / (dst_h / 2.0);
    double dst_vertical_perpixel_y = vertical_dst_y / (dst_h / 2.0);
    double dst_vertical_perpixel_z = vertical_dst_z / (dst_h / 2.0);

    //The size of the per-pixel displacement.
    double dst_horizontal_perpixel_size = sqrt((dst_horizontal_perpixel_x * dst_horizontal_perpixel_x) + (dst_horizontal_perpixel_y * dst_horizontal_perpixel_y) + (dst_horizontal_perpixel_z * dst_horizontal_perpixel_z));
    double dst_vertical_perpixel_size = sqrt((dst_vertical_perpixel_x * dst_vertical_perpixel_x) + (dst_vertical_perpixel_y * dst_vertical_perpixel_y) + (dst_vertical_perpixel_z * dst_vertical_perpixel_z));

    pixel point;
    int src_midw = src_w / 2;
    int src_midh = src_h / 2;
    int dst_midw = dst_w / 2;
    int dst_midh = dst_h / 2;
    double pos_x, pos_y, pos_z, delta_x, delta_y, delta_z, dst_x, dst_y, dst_z, distance, depth3D;
    double a[3][3], b[3], c[3], size, vx, vy, vz;
    int i1, j1;
    double Y_err = 0, R, G, B, Y1, U1, V1, Y2, U2, V2;
    int Y_cnt = 0;
    //    bool changed[H][W];
    //    memset(changed, false, sizeof(changed));
    int change_cnt = 0;
    int perspective = 0, cover = 0;
    double max_err = -1, err;
    for (int i = start_h; i < end_h; ++i)
        for (int j = start_w; j < end_w; ++j) {
            //computing the coordinate of 3D point for each pixel in the original frame.
            delta_x = src_viewpoint_dx + (0.5 + j - src_midw) * src_horizontal_perpixel_x + (0.5 + i - src_midh) * src_vertical_perpixel_x;
            delta_y = src_viewpoint_dy + (0.5 + j - src_midw) * src_horizontal_perpixel_y + (0.5 + i - src_midh) * src_vertical_perpixel_y;
            delta_z = src_viewpoint_dz + (0.5 + j - src_midw) * src_horizontal_perpixel_z + (0.5 + i - src_midh) * src_vertical_perpixel_z;

            size = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
            depth3D = src[i - start_h][j - start_w].depth * size / distance_to_plane;

            //delta is the displacement from the source viewpoint to the 3D position of the pixel (i, j).
            delta_x = delta_x / size * depth3D;
            delta_y = delta_y / size * depth3D;
            delta_z = delta_z / size * depth3D;
            //pos is the 3D position of pixel (i, j).
            pos_x = src_viewpoint.x + delta_x;
            pos_y = src_viewpoint.y + delta_y;
            pos_z = src_viewpoint.z + delta_z;

            //matrix a is the arguments of Linear simultaneous equations in three unknowns
            a[0][0] = pos_y - dst_viewpoint.y;
            a[0][1] = dst_viewpoint.x - pos_x;
            a[0][2] = 0;
            b[0] = -((dst_viewpoint.y - pos_y) * dst_viewpoint.x + (pos_x - dst_viewpoint.x) * dst_viewpoint.y);

            a[1][0] = pos_z - dst_viewpoint.z;
            a[1][1] = 0;
            a[1][2] = dst_viewpoint.x - pos_x;
            b[1] = -((dst_viewpoint.z - pos_z) * dst_viewpoint.x + (pos_x - dst_viewpoint.x) * dst_viewpoint.z);

            a[2][0] = dst_viewpoint_dx;
            a[2][1] = dst_viewpoint_dy;
            a[2][2] = dst_viewpoint_dz;
            b[2] = -(-(dst_viewpoint.x + dst_viewpoint_dx) * dst_viewpoint_dx
                - (dst_viewpoint.y + dst_viewpoint_dy) * dst_viewpoint_dy
                - (dst_viewpoint.z + dst_viewpoint_dz) * dst_viewpoint_dz);

            gaussin(a, b, c);
            dst_x = c[0], dst_y = c[1], dst_z = c[2];
            distance = (pos_x - dst_viewpoint.x) * (pos_x - dst_viewpoint.x) + (pos_y - dst_viewpoint.y) * (pos_y - dst_viewpoint.y) + (pos_z - dst_viewpoint.z) * (pos_z - dst_viewpoint.z);
            if ((dst_x - dst_viewpoint.x) * (pos_x - dst_viewpoint.x) + (dst_y - dst_viewpoint.y) * (pos_y - dst_viewpoint.y) + (dst_z - dst_viewpoint.z) * (pos_z - dst_viewpoint.z) < 0) continue;

            //(vx, vy, vz) is the displacement of the point (dst_x, dst_y, dst_z) from the central point of the dstination frame.
            vx = dst_x - (dst_viewpoint.x + dst_viewpoint_dx);
            vy = dst_y - (dst_viewpoint.y + dst_viewpoint_dy);
            vz = dst_z - (dst_viewpoint.z + dst_viewpoint_dz);

            //(i1, j1) is the position in the destination frame of the pixel (i, j) in the source frame.
            i1 = dst_midh + round(-0.5 + (vx * vertical_dst_x + vy * vertical_dst_y + vz * vertical_dst_z) / dst_distance_vertical / dst_vertical_perpixel_size);
            j1 = dst_midw + round(-0.5 + (vx * horizontal_dst_x + vy * horizontal_dst_y + vz * horizontal_dst_z) / dst_distance_horizontal / dst_horizontal_perpixel_size);

            //Copy the pixel value of (i, j) in the source frame to (i1, j1) in the destination frame.
            if (j1 >= 0 && j1 < dst_w && i1 >= 0 && i1 < dst_h) {
                if (dst[i1][j1].valid == 1 && dst[i1][j1].human == 0) {
                    cover++;
                    err = sqrt(distance) - sqrt(dst[i1][j1].depth);
                    max_err = max(err, max_err);
                    if (err > 0) {
                        perspective++;
                    }
                }
                if (dst[i1][j1].valid == 0 || timestamp > dst[i1][j1].timestamp || (timestamp == dst[i1][j1].timestamp && sqrt(dst[i1][j1].depth) - sqrt(distance) > 0.3)) {
                    dst[i1][j1].r = src[i - start_h][j - start_w].r;
                    dst[i1][j1].g = src[i - start_h][j - start_w].g;
                    dst[i1][j1].b = src[i - start_h][j - start_w].b;
                    dst[i1][j1].valid = 1;
                    dst[i1][j1].human |= src[i - start_h][j - start_w].human;
                    dst[i1][j1].depth = distance;
                    dst[i1][j1].timestamp = timestamp;
                    dst_location[i1][j1].x = pos_x;
                    dst_location[i1][j1].y = pos_y;
                    dst_location[i1][j1].z = pos_z;
                    if (changed[i1][j1] == false) {
                        changed[i1][j1] = true;
                        change_h[change_cnt] = i1;
                        change_w[change_cnt] = j1;
                        change_cnt++;
                    }
                }
                else if (dst[i1][j1].valid == 1 && dst[i1][j1].human == 0 && !changed[i1][j1]) {
                    get_YUV(dst[i1][j1], R, G, B, Y1, U1, V1);
                    get_YUV(src[i - start_h][j - start_w], R, G, B, Y2, U2, V2);
                    if (abs(sqrt(dst[i1][j1].depth) - sqrt(distance)) < 0.04
                        /*&& abs(U1 - U2) < 3 && abs(V1 - V2) < 3*/) {
                        Y_err += (Y1 - Y2);
                        Y_cnt++;
                        changed[i1][j1] = true;
                    }
                }
            }
        }

    if (max_err > 100000 || cover == 0 || Y_cnt == 0/* || 1.0*perspective/cover > 0.2*/) {
        for (int i = 0; i < change_cnt; i++) {
            dst[change_h[i]][change_w[i]].valid = 0;
        }
        return 0;
    }
    if (Y_cnt > 0) {
        Y_err = Y_err / Y_cnt;
        double b_sum = 0, i_sum = 0;
        int psnr_cnt = 0;

        for (int i = 0; i < change_cnt; i++) {
            int tmp_h = change_h[i];
            int tmp_w = change_w[i];
            add_Y_channel(dst[tmp_h][tmp_w], Y_err);
            if (mask[tmp_h / 2][tmp_w / 2] != 1 && dst[tmp_h][tmp_w].human % 2 == 0) {
                uchar* ptr_gt = MatGT.ptr<uchar>(tmp_h / 2, tmp_w / 2);

                b_sum += (ptr_gt[0] - dst[tmp_h][tmp_w].b) * (ptr_gt[0] - dst[tmp_h][tmp_w].b);
                b_sum += (ptr_gt[1] - dst[tmp_h][tmp_w].g) * (ptr_gt[1] - dst[tmp_h][tmp_w].g);
                b_sum += (ptr_gt[2] - dst[tmp_h][tmp_w].r) * (ptr_gt[2] - dst[tmp_h][tmp_w].r);
                
                i_sum += (ptr_gt[0] - dst_inpainting[tmp_h][tmp_w].b) * (ptr_gt[0] - dst_inpainting[tmp_h][tmp_w].b);
                i_sum += (ptr_gt[1] - dst_inpainting[tmp_h][tmp_w].g) * (ptr_gt[1] - dst_inpainting[tmp_h][tmp_w].g);
                i_sum += (ptr_gt[2] - dst_inpainting[tmp_h][tmp_w].r) * (ptr_gt[2] - dst_inpainting[tmp_h][tmp_w].r);

                psnr_cnt += 3;
            }
        }
        if (psnr_cnt > 0 && b_sum > 0 && i_sum > 0) {
            b_sum = b_sum / (psnr_cnt * 1.0);
            i_sum = i_sum / (psnr_cnt * 1.0);

            psnr_block = 10.0 * log10((255 * 255) / b_sum);
            psnr_inpaint = 10.0 * log10((255 * 255) / i_sum);
        }
        else {
            psnr_block = 0;
            psnr_inpaint = 0;
        }
    }
	return change_cnt;
}


double image_warping::block_super_resolution_estimation(
    int src_w, int src_h,
    int dst_w, int dst_h,
    int start_w, int end_w, int start_h, int end_h,
    double src_wFoV, double src_hFoV,
    double dst_wFoV, double dst_hFoV,
    pixel src[H][W],
    viewpoint src_viewpoint, viewpoint dst_viewpoint,
    double distance_to_plane
)
{
    //These are temperal variations used for vector rotation (by calling the function rotate()).
    double dst_vector[3], src_vector[3], Euler[3];

    //Euler angles of the source viewpoint.
    Euler[0] = src_viewpoint.pitch;
    Euler[1] = src_viewpoint.roll;
    Euler[2] = src_viewpoint.yaw;

    //Computing the direction vector of the source viewpoint.
    src_vector[0] = 0, src_vector[1] = 1, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double src_viewpoint_dx = dst_vector[0];
    double src_viewpoint_dy = dst_vector[1];
    double src_viewpoint_dz = dst_vector[2];

    //Computing the horizontal vector of the source viewpoint.
    src_vector[0] = 1, src_vector[1] = 0, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double horizontal_src_x = dst_vector[0];
    double horizontal_src_y = dst_vector[1];
    double horizontal_src_z = dst_vector[2];

    //Computing the vertical vector of the source viewpoint.
    src_vector[0] = 0, src_vector[1] = 0, src_vector[2] = -1;
    rotate(dst_vector, Euler, src_vector);
    double vertical_src_x = dst_vector[0];
    double vertical_src_y = dst_vector[1];
    double vertical_src_z = dst_vector[2];

    //Euler angles of the destination viewpoint.
    Euler[0] = dst_viewpoint.pitch;
    Euler[1] = dst_viewpoint.roll;
    Euler[2] = dst_viewpoint.yaw;

    //Computing the direction vector of the destination viewpoint.
    src_vector[0] = 0, src_vector[1] = 1, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double dst_viewpoint_dx = dst_vector[0];
    double dst_viewpoint_dy = dst_vector[1];
    double dst_viewpoint_dz = dst_vector[2];

    //Computing the horizontal vector of the destination viewpoint.
    src_vector[0] = 1, src_vector[1] = 0, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double horizontal_dst_x = dst_vector[0];
    double horizontal_dst_y = dst_vector[1];
    double horizontal_dst_z = dst_vector[2];

    //Computing the vertical vector of the destination viewpoint.
    src_vector[0] = 0, src_vector[1] = 0, src_vector[2] = -1;
    rotate(dst_vector, Euler, src_vector);
    double vertical_dst_x = dst_vector[0];
    double vertical_dst_y = dst_vector[1];
    double vertical_dst_z = dst_vector[2];

    //normalization the vectors based on the value of distance_to_plane.
    src_viewpoint_dx *= distance_to_plane;
    src_viewpoint_dy *= distance_to_plane;
    src_viewpoint_dz *= distance_to_plane;
    dst_viewpoint_dx *= distance_to_plane;
    dst_viewpoint_dy *= distance_to_plane;
    dst_viewpoint_dz *= distance_to_plane;

    //computing the size of the source frame in 3D space.
    double src_distance_vertical = tan(src_hFoV / 180.0 * pi / 2) * distance_to_plane;
    double src_distance_horizontal = tan(src_wFoV / 180.0 * pi / 2) * distance_to_plane;
    double dst_distance_vertical = tan(dst_hFoV / 180.0 * pi / 2) * distance_to_plane;
    double dst_distance_horizontal = tan(dst_wFoV / 180.0 * pi / 2) * distance_to_plane;

    horizontal_src_x *= src_distance_horizontal;
    horizontal_src_y *= src_distance_horizontal;
    horizontal_src_z *= src_distance_horizontal;

    horizontal_dst_x *= dst_distance_horizontal;
    horizontal_dst_y *= dst_distance_horizontal;
    horizontal_dst_z *= dst_distance_horizontal;

    vertical_src_x *= src_distance_vertical;
    vertical_src_y *= src_distance_vertical;
    vertical_src_z *= src_distance_vertical;

    vertical_dst_x *= dst_distance_vertical;
    vertical_dst_y *= dst_distance_vertical;
    vertical_dst_z *= dst_distance_vertical;

    double src_horizontal_perpixel_x = horizontal_src_x / (src_w / 2.0);
    double src_horizontal_perpixel_y = horizontal_src_y / (src_w / 2.0);
    double src_horizontal_perpixel_z = horizontal_src_z / (src_w / 2.0);
    double src_vertical_perpixel_x = vertical_src_x / (src_h / 2.0);
    double src_vertical_perpixel_y = vertical_src_y / (src_h / 2.0);
    double src_vertical_perpixel_z = vertical_src_z / (src_h / 2.0);

    //Computing the per-pixel displacement in 3D space from the central point of the frame (both horizontal and vertical).
    double dst_horizontal_perpixel_x = horizontal_dst_x / (dst_w / 2.0);
    double dst_horizontal_perpixel_y = horizontal_dst_y / (dst_w / 2.0);
    double dst_horizontal_perpixel_z = horizontal_dst_z / (dst_w / 2.0);
    double dst_vertical_perpixel_x = vertical_dst_x / (dst_h / 2.0);
    double dst_vertical_perpixel_y = vertical_dst_y / (dst_h / 2.0);
    double dst_vertical_perpixel_z = vertical_dst_z / (dst_h / 2.0);

    //The size of the per-pixel displacement.
    double dst_horizontal_perpixel_size = sqrt((dst_horizontal_perpixel_x * dst_horizontal_perpixel_x) + (dst_horizontal_perpixel_y * dst_horizontal_perpixel_y) + (dst_horizontal_perpixel_z * dst_horizontal_perpixel_z));
    double dst_vertical_perpixel_size = sqrt((dst_vertical_perpixel_x * dst_vertical_perpixel_x) + (dst_vertical_perpixel_y * dst_vertical_perpixel_y) + (dst_vertical_perpixel_z * dst_vertical_perpixel_z));

    pixel point;
    int src_midw = src_w / 2;
    int src_midh = src_h / 2;
    int dst_midw = dst_w / 2;
    int dst_midh = dst_h / 2;
    double pos_x, pos_y, pos_z, delta_x, delta_y, delta_z, dst_x, dst_y, dst_z, distance, depth3D;
    double a[3][3], b[3], c[3], size, vx, vy, vz;
    int i1, j1, i, j;

    int corner[5][2] = { {start_h, start_w} , {start_h, end_w - 1} , {end_h - 1, end_w - 1}, {end_h - 1, start_w} ,  {start_h, start_w} };
    int x_corner[5], y_corner[5];

    for (int k = 0; k < 5; ++k) {
        i = corner[k][0], j = corner[k][1];
        //computing the coordinate of 3D point for each pixel in the original frame.
        delta_x = src_viewpoint_dx + (0.5 + j - src_midw) * src_horizontal_perpixel_x + (0.5 + i - src_midh) * src_vertical_perpixel_x;
        delta_y = src_viewpoint_dy + (0.5 + j - src_midw) * src_horizontal_perpixel_y + (0.5 + i - src_midh) * src_vertical_perpixel_y;
        delta_z = src_viewpoint_dz + (0.5 + j - src_midw) * src_horizontal_perpixel_z + (0.5 + i - src_midh) * src_vertical_perpixel_z;

        size = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
        depth3D = src[i - start_h][j - start_w].depth * size / distance_to_plane;

        //delta is the displacement from the source viewpoint to the 3D position of the pixel (i, j).
        delta_x = delta_x / size * depth3D;
        delta_y = delta_y / size * depth3D;
        delta_z = delta_z / size * depth3D;
        //pos is the 3D position of pixel (i, j).
        pos_x = src_viewpoint.x + delta_x;
        pos_y = src_viewpoint.y + delta_y;
        pos_z = src_viewpoint.z + delta_z;

        //matrix a is the arguments of Linear simultaneous equations in three unknowns
        a[0][0] = pos_y - dst_viewpoint.y;
        a[0][1] = dst_viewpoint.x - pos_x;
        a[0][2] = 0;
        b[0] = -((dst_viewpoint.y - pos_y) * dst_viewpoint.x + (pos_x - dst_viewpoint.x) * dst_viewpoint.y);

        a[1][0] = pos_z - dst_viewpoint.z;
        a[1][1] = 0;
        a[1][2] = dst_viewpoint.x - pos_x;
        b[1] = -((dst_viewpoint.z - pos_z) * dst_viewpoint.x + (pos_x - dst_viewpoint.x) * dst_viewpoint.z);

        a[2][0] = dst_viewpoint_dx;
        a[2][1] = dst_viewpoint_dy;
        a[2][2] = dst_viewpoint_dz;
        b[2] = -(-(dst_viewpoint.x + dst_viewpoint_dx) * dst_viewpoint_dx
            - (dst_viewpoint.y + dst_viewpoint_dy) * dst_viewpoint_dy
            - (dst_viewpoint.z + dst_viewpoint_dz) * dst_viewpoint_dz);

        gaussin(a, b, c);
        dst_x = c[0], dst_y = c[1], dst_z = c[2];
        distance = (pos_x - dst_viewpoint.x) * (pos_x - dst_viewpoint.x) + (pos_y - dst_viewpoint.y) * (pos_y - dst_viewpoint.y) + (pos_z - dst_viewpoint.z) * (pos_z - dst_viewpoint.z);
        //if ((dst_x - dst_viewpoint.x) * (pos_x - dst_viewpoint.x) + (dst_y - dst_viewpoint.y) * (pos_y - dst_viewpoint.y) + (dst_z - dst_viewpoint.z) * (pos_z - dst_viewpoint.z) < 0) continue;

        //(vx, vy, vz) is the displacement of the point (dst_x, dst_y, dst_z) from the central point of the dstination frame.
        vx = dst_x - (dst_viewpoint.x + dst_viewpoint_dx);
        vy = dst_y - (dst_viewpoint.y + dst_viewpoint_dy);
        vz = dst_z - (dst_viewpoint.z + dst_viewpoint_dz);

        //(i1, j1) is the position in the destination frame of the pixel (i, j) in the source frame.
        x_corner[k] = dst_midh + round(-0.5 + (vx * vertical_dst_x + vy * vertical_dst_y + vz * vertical_dst_z) / dst_distance_vertical / dst_vertical_perpixel_size);
        y_corner[k] = dst_midw + round(-0.5 + (vx * horizontal_dst_x + vy * horizontal_dst_y + vz * horizontal_dst_z) / dst_distance_horizontal / dst_horizontal_perpixel_size);
    }

    //    for (int i = 0; i < 4; ++i)
    //        printf("%d %d\n", x_corner[i], y_corner[i]);
    //    cout<<endl;
    double max_edge = 0, min_edge;
    for (int i = 0; i < 4; ++i)
        max_edge = max(max_edge, sqrt(double((x_corner[i] - x_corner[i + 1]) * (x_corner[i] - x_corner[i + 1]) + (y_corner[i] - y_corner[i + 1]) * (y_corner[i] - y_corner[i + 1]))));
    min_edge = (double)min(end_w - start_w, end_h - start_h);
    double radio = max_edge / min_edge;
    //if (radio < 1) radio = 1;
    return radio;
}

void image_warping::block_corner_estimation(
    int src_w, int src_h,
    int dst_w, int dst_h,
    int start_w, int end_w, int start_h, int end_h,
    double src_wFoV, double src_hFoV,
    double dst_wFoV, double dst_hFoV,
    pixel src[H][W],
    viewpoint src_viewpoint, viewpoint dst_viewpoint,
    double distance_to_plane,
    int* corner_2D
)
{
    //These are temperal variations used for vector rotation (by calling the function rotate()).
    double dst_vector[3], src_vector[3], Euler[3];

    //Euler angles of the source viewpoint.
    Euler[0] = src_viewpoint.pitch;
    Euler[1] = src_viewpoint.roll;
    Euler[2] = src_viewpoint.yaw;

    //Computing the direction vector of the source viewpoint.
    src_vector[0] = 0, src_vector[1] = 1, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double src_viewpoint_dx = dst_vector[0];
    double src_viewpoint_dy = dst_vector[1];
    double src_viewpoint_dz = dst_vector[2];

    //Computing the horizontal vector of the source viewpoint.
    src_vector[0] = 1, src_vector[1] = 0, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double horizontal_src_x = dst_vector[0];
    double horizontal_src_y = dst_vector[1];
    double horizontal_src_z = dst_vector[2];

    //Computing the vertical vector of the source viewpoint.
    src_vector[0] = 0, src_vector[1] = 0, src_vector[2] = -1;
    rotate(dst_vector, Euler, src_vector);
    double vertical_src_x = dst_vector[0];
    double vertical_src_y = dst_vector[1];
    double vertical_src_z = dst_vector[2];

    //Euler angles of the destination viewpoint.
    Euler[0] = dst_viewpoint.pitch;
    Euler[1] = dst_viewpoint.roll;
    Euler[2] = dst_viewpoint.yaw;

    //Computing the direction vector of the destination viewpoint.
    src_vector[0] = 0, src_vector[1] = 1, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double dst_viewpoint_dx = dst_vector[0];
    double dst_viewpoint_dy = dst_vector[1];
    double dst_viewpoint_dz = dst_vector[2];

    //Computing the horizontal vector of the destination viewpoint.
    src_vector[0] = 1, src_vector[1] = 0, src_vector[2] = 0;
    rotate(dst_vector, Euler, src_vector);
    double horizontal_dst_x = dst_vector[0];
    double horizontal_dst_y = dst_vector[1];
    double horizontal_dst_z = dst_vector[2];

    //Computing the vertical vector of the destination viewpoint.
    src_vector[0] = 0, src_vector[1] = 0, src_vector[2] = -1;
    rotate(dst_vector, Euler, src_vector);
    double vertical_dst_x = dst_vector[0];
    double vertical_dst_y = dst_vector[1];
    double vertical_dst_z = dst_vector[2];

    //normalization the vectors based on the value of distance_to_plane.
    src_viewpoint_dx *= distance_to_plane;
    src_viewpoint_dy *= distance_to_plane;
    src_viewpoint_dz *= distance_to_plane;
    dst_viewpoint_dx *= distance_to_plane;
    dst_viewpoint_dy *= distance_to_plane;
    dst_viewpoint_dz *= distance_to_plane;

    //computing the size of the source frame in 3D space.
    double src_distance_vertical = tan(src_hFoV / 180.0 * pi / 2) * distance_to_plane;
    double src_distance_horizontal = tan(src_wFoV / 180.0 * pi / 2) * distance_to_plane;
    double dst_distance_vertical = tan(dst_hFoV / 180.0 * pi / 2) * distance_to_plane;
    double dst_distance_horizontal = tan(dst_wFoV / 180.0 * pi / 2) * distance_to_plane;

    horizontal_src_x *= src_distance_horizontal;
    horizontal_src_y *= src_distance_horizontal;
    horizontal_src_z *= src_distance_horizontal;

    horizontal_dst_x *= dst_distance_horizontal;
    horizontal_dst_y *= dst_distance_horizontal;
    horizontal_dst_z *= dst_distance_horizontal;

    vertical_src_x *= src_distance_vertical;
    vertical_src_y *= src_distance_vertical;
    vertical_src_z *= src_distance_vertical;

    vertical_dst_x *= dst_distance_vertical;
    vertical_dst_y *= dst_distance_vertical;
    vertical_dst_z *= dst_distance_vertical;

    double src_horizontal_perpixel_x = horizontal_src_x / (src_w / 2.0);
    double src_horizontal_perpixel_y = horizontal_src_y / (src_w / 2.0);
    double src_horizontal_perpixel_z = horizontal_src_z / (src_w / 2.0);
    double src_vertical_perpixel_x = vertical_src_x / (src_h / 2.0);
    double src_vertical_perpixel_y = vertical_src_y / (src_h / 2.0);
    double src_vertical_perpixel_z = vertical_src_z / (src_h / 2.0);

    //Computing the per-pixel displacement in 3D space from the central point of the frame (both horizontal and vertical).
    double dst_horizontal_perpixel_x = horizontal_dst_x / (dst_w / 2.0);
    double dst_horizontal_perpixel_y = horizontal_dst_y / (dst_w / 2.0);
    double dst_horizontal_perpixel_z = horizontal_dst_z / (dst_w / 2.0);
    double dst_vertical_perpixel_x = vertical_dst_x / (dst_h / 2.0);
    double dst_vertical_perpixel_y = vertical_dst_y / (dst_h / 2.0);
    double dst_vertical_perpixel_z = vertical_dst_z / (dst_h / 2.0);

    //The size of the per-pixel displacement.
    double dst_horizontal_perpixel_size = sqrt((dst_horizontal_perpixel_x * dst_horizontal_perpixel_x) + (dst_horizontal_perpixel_y * dst_horizontal_perpixel_y) + (dst_horizontal_perpixel_z * dst_horizontal_perpixel_z));
    double dst_vertical_perpixel_size = sqrt((dst_vertical_perpixel_x * dst_vertical_perpixel_x) + (dst_vertical_perpixel_y * dst_vertical_perpixel_y) + (dst_vertical_perpixel_z * dst_vertical_perpixel_z));

    pixel point;
    int src_midw = src_w / 2;
    int src_midh = src_h / 2;
    int dst_midw = dst_w / 2;
    int dst_midh = dst_h / 2;
    double pos_x, pos_y, pos_z, delta_x, delta_y, delta_z, dst_x, dst_y, dst_z, distance, depth3D;
    double a[3][3], b[3], c[3], size, vx, vy, vz;
    int i1, j1, i, j;

    int corner[5][2] = { {start_h, start_w} , {start_h, end_w - 1} , {end_h - 1, end_w - 1}, {end_h - 1, start_w} ,  {start_h, start_w} };
    //corner_2D = new int[10];

    for (int k = 0; k < 4; ++k) {
        i = corner[k][0], j = corner[k][1];
        //computing the coordinate of 3D point for each pixel in the original frame.
        delta_x = src_viewpoint_dx + (0.5 + j - src_midw) * src_horizontal_perpixel_x + (0.5 + i - src_midh) * src_vertical_perpixel_x;
        delta_y = src_viewpoint_dy + (0.5 + j - src_midw) * src_horizontal_perpixel_y + (0.5 + i - src_midh) * src_vertical_perpixel_y;
        delta_z = src_viewpoint_dz + (0.5 + j - src_midw) * src_horizontal_perpixel_z + (0.5 + i - src_midh) * src_vertical_perpixel_z;

        size = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
        depth3D = src[i - start_h][j - start_w].depth * size / distance_to_plane;

        //delta is the displacement from the source viewpoint to the 3D position of the pixel (i, j).
        delta_x = delta_x / size * depth3D;
        delta_y = delta_y / size * depth3D;
        delta_z = delta_z / size * depth3D;
        //pos is the 3D position of pixel (i, j).
        pos_x = src_viewpoint.x + delta_x;
        pos_y = src_viewpoint.y + delta_y;
        pos_z = src_viewpoint.z + delta_z;

        //matrix a is the arguments of Linear simultaneous equations in three unknowns
        a[0][0] = pos_y - dst_viewpoint.y;
        a[0][1] = dst_viewpoint.x - pos_x;
        a[0][2] = 0;
        b[0] = -((dst_viewpoint.y - pos_y) * dst_viewpoint.x + (pos_x - dst_viewpoint.x) * dst_viewpoint.y);

        a[1][0] = pos_z - dst_viewpoint.z;
        a[1][1] = 0;
        a[1][2] = dst_viewpoint.x - pos_x;
        b[1] = -((dst_viewpoint.z - pos_z) * dst_viewpoint.x + (pos_x - dst_viewpoint.x) * dst_viewpoint.z);

        a[2][0] = dst_viewpoint_dx;
        a[2][1] = dst_viewpoint_dy;
        a[2][2] = dst_viewpoint_dz;
        b[2] = -(-(dst_viewpoint.x + dst_viewpoint_dx) * dst_viewpoint_dx
            - (dst_viewpoint.y + dst_viewpoint_dy) * dst_viewpoint_dy
            - (dst_viewpoint.z + dst_viewpoint_dz) * dst_viewpoint_dz);

        gaussin(a, b, c);
        dst_x = c[0], dst_y = c[1], dst_z = c[2];
        distance = (pos_x - dst_viewpoint.x) * (pos_x - dst_viewpoint.x) + (pos_y - dst_viewpoint.y) * (pos_y - dst_viewpoint.y) + (pos_z - dst_viewpoint.z) * (pos_z - dst_viewpoint.z);
        //if ((dst_x - dst_viewpoint.x) * (pos_x - dst_viewpoint.x) + (dst_y - dst_viewpoint.y) * (pos_y - dst_viewpoint.y) + (dst_z - dst_viewpoint.z) * (pos_z - dst_viewpoint.z) < 0) continue;

        //(vx, vy, vz) is the displacement of the point (dst_x, dst_y, dst_z) from the central point of the dstination frame.
        vx = dst_x - (dst_viewpoint.x + dst_viewpoint_dx);
        vy = dst_y - (dst_viewpoint.y + dst_viewpoint_dy);
        vz = dst_z - (dst_viewpoint.z + dst_viewpoint_dz);

        //(i1, j1) is the position in the destination frame of the pixel (i, j) in the source frame.
        corner_2D[k] = dst_midh + round(-0.5 + (vx * vertical_dst_x + vy * vertical_dst_y + vz * vertical_dst_z) / dst_distance_vertical / dst_vertical_perpixel_size);
        corner_2D[k + 4] = dst_midw + round(-0.5 + (vx * horizontal_dst_x + vy * horizontal_dst_y + vz * horizontal_dst_z) / dst_distance_horizontal / dst_horizontal_perpixel_size);
    }
}