#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <iomanip>
#include <fstream>
#include <stdint.h>
#include <bitset>
#include <math.h>
#include <ctime>
#include <direct.h>
#include <io.h>

#include <cstring>
#include "multi_image_warping.cpp"
#include "KDtree_v2.h"

#define FRAME_CNT 10
#define MAXN 100

#pragma warning(disable : 4996)

using namespace cv;
using std::cout; using std::cerr; using std::endl;

const int CNT = 10000;
int frame_number = CNT, cur_frame_identity = -1;
viewpoint view_point[CNT];
double HFOV[CNT];
double WFOV[CNT];

image_warping inst;
KD_tree tree;
pixel src[H_GT][W_GT];
pixel src_up[H][W];
pixel src_block[H][W];
pixel src_block_up[H][W];
pixel dst_up[H][W];
pixel dst_inpainting[H][W];
Location_3D dst_location[H][W], src_location[H][W];

int inpainting_dis = 6;
int downsampling_ratio = 64;
bool used_p[H][W];
bool changed[H][W];
int up_size; int inpainting_size;
int change_h[W * H];
int change_w[W * H];
Mat MatGT;
KD_tree::ele* chosen_e;

int mask_real[H_GT][W_GT];
int mask_fake[H_GT][W_GT];
int mask_ref[H_GT][W_GT];

string abs_dir;
string seq_name;
string outpath;
string inpath;

double cam_near_clip = 0.15;
double cam_far_clip = 500;
int human_id;

ofstream output_log;

void getMask(string path, int mask[H_GT][W_GT]) {
	memset(mask[0], 0, H_GT * sizeof(mask[0]));

	Mat Mat_mask = imread(path.data(), 2);

	if (Mat_mask.data == NULL) {
		puts("No Image!");
		return;
	}

	int mask_tmp;
	for (int h = 0; h < Mat_mask.rows; ++h)
	{
		for (int w = 0; w < Mat_mask.cols; ++w)
		{
			uchar* ptr = Mat_mask.ptr<uchar>(h, w);
			mask_tmp = ptr[0];
			if (mask_tmp == human_id)
				mask_tmp = 1;
			else if (mask_tmp != 0)
				mask_tmp = 2;
			else
				mask_tmp = 0;
			if (mask_tmp != 0) {
				for (int i = -1; i <= 1; i++)
					for (int j = -1; j <= 1; j++)
						if (h + i >= 0 && h + i < H_GT && w + j >= 0 && w + j < W_GT)
							mask[h + i][w + j] = mask_tmp;
			}
		}
	}
}

void read_viewpoint_euler(string path) {
	int idx;
	FILE* inf;
	inf = fopen(path.data(), "r");
	for (int i = 0; i < frame_number; ++i) {
		fscanf(inf, "%d%lf%lf%lf%lf%lf%lf%lf", &idx, &view_point[i].x, &view_point[i].y, &view_point[i].z, &HFOV[i], &view_point[i].pitch, &view_point[i].roll, &view_point[i].yaw);
		WFOV[i] = atan(tan(HFOV[i] / 180.0 * pi / 2) * 1920 / 1080.0) / pi * 180 * 2;
	}
	fclose(inf);
}


double getPSNR(const Mat& I1, const Mat& I2, int cnt)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);         // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double  mse = sse / (double)(I1.channels() * (I1.total() - cnt));
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}

void getDepth(string path, int flag_up_down) {
	uint32_t depth_uint32;
	double depth_float;

	uint8_t val[4] = { 0, 0, 0, 0 };
	Mat Mat_depth = imread(path.data(), 1);
	if (Mat_depth.data == NULL) {
		puts("No Image!");
		return;
	}
	for (int h = 0; h < Mat_depth.rows; ++h)
	{
		for (int w = 0; w < Mat_depth.cols; ++w)
		{
			uchar* ptr_depth = Mat_depth.ptr<uchar>(h, w);
			val[0] = ptr_depth[0];
			val[1] = ptr_depth[1];
			val[2] = ptr_depth[2];
			memcpy(&depth_uint32, val, sizeof(uint32_t));
			depth_float = (double)depth_uint32;
			depth_float = 0.05 * 1000.0 / depth_float;
			depth_float = (cam_near_clip * cam_far_clip / (cam_near_clip + depth_float * (cam_far_clip - cam_near_clip)));
			if (depth_float == 0)
				depth_float = 1000;
			depth_float = (double)((int)(depth_float * 1000)) / 1000;
			src[h][w].depth = depth_float;
		}
	}

	if (flag_up_down) {
		double step_i, step_j;
		int next_i, next_j;
		for (int i = 0; i < Mat_depth.rows; ++i) {
			for (int j = 0; j < Mat_depth.cols; ++j) {
				if (i + 1 < Mat_depth.rows)
					step_i = (src[i + 1][j].depth - src[i][j].depth);
				else
					step_i = 0;

				if (j + 1 < Mat_depth.cols)
					step_j = (src[i][j + 1].depth - src[i][j].depth);
				else
					step_j = 0;

				if (step_i > 0.5 || step_i < -0.5) step_i = 0;
				if (step_j > 0.5 || step_j < -0.5) step_j = 0;
				step_i = step_i / 2;
				step_j = step_j / 2;
				for (int up_i = 0; up_i < 2; up_i++)
					for (int up_j = 0; up_j < 2; up_j++) {
						next_i = i * 2 + up_i;
						next_j = j * 2 + up_j;
						if (next_i < H && next_j < W) {
							src_up[next_i][next_j].depth = src[i][j].depth + step_i * up_i + step_j * up_j;
						}
					}
			}
		}
	}
}

void getSrc(string path, int flag_up_down) {
	Mat MatSRC = imread(path.data(), 1);
	if (MatSRC.data == NULL) {
		puts("No Image!");
		return;
	}
	for (int h = 0; h < MatSRC.rows; ++h)
	{
		for (int w = 0; w < MatSRC.cols; ++w)
		{
			uchar* ptr = MatSRC.ptr<uchar>(h, w);
			src[h][w].b = ptr[0];
			src[h][w].g = ptr[1];
			src[h][w].r = ptr[2];
			src[h][w].human = mask_ref[h][w];

			if (flag_up_down) {
				src_up[2 * h][2 * w].b = ptr[0];
				src_up[2 * h][2 * w + 1].b = ptr[0];
				src_up[2 * h + 1][2 * w].b = ptr[0];
				src_up[2 * h + 1][2 * w + 1].b = ptr[0];

				src_up[2 * h][2 * w].g = ptr[1];
				src_up[2 * h][2 * w + 1].g = ptr[1];
				src_up[2 * h + 1][2 * w].g = ptr[1];
				src_up[2 * h + 1][2 * w + 1].g = ptr[1];

				src_up[2 * h][2 * w].r = ptr[2];
				src_up[2 * h][2 * w + 1].r = ptr[2];
				src_up[2 * h + 1][2 * w].r = ptr[2];
				src_up[2 * h + 1][2 * w + 1].r = ptr[2];

				src_up[2 * h][2 * w].human = mask_ref[h][w];
				src_up[2 * h][2 * w + 1].human = mask_ref[h][w];
				src_up[2 * h + 1][2 * w].human = mask_ref[h][w];
				src_up[2 * h + 1][2 * w + 1].human = mask_ref[h][w];
			}
		}
	}
}

void copy_mat(pixel from[H][W], pixel to[H][W]) {
	for (int h = 0; h < H; h++)
		for (int w = 0; w < W; w++) {
			to[h][w].b = from[h][w].b;
			to[h][w].g = from[h][w].g;
			to[h][w].r = from[h][w].r;
			to[h][w].depth = from[h][w].depth;
			to[h][w].human = from[h][w].human;
			to[h][w].timestamp = from[h][w].timestamp;
			to[h][w].valid = from[h][w].valid;
		}
}

void prepare(int frame, int flag_mask, int flag_up_down) {
	string id = to_string(frame);
	stringstream ss;
	ss << setw(5) << setfill('0') << frame;
	string img_path = inpath + ss.str() + ".jpg";
	string depth_path = inpath + ss.str() + ".png";
	string mask_path = inpath + ss.str() + "_id.png";
	if (flag_mask)
		getMask(mask_path, mask_ref);
	getDepth(depth_path, flag_up_down);
	getSrc(img_path, flag_up_down);
}


void get_DST(Mat& MatDST) {
	for (int h = 0; h < MatDST.rows; ++h) {
		for (int w = 0; w < MatDST.cols; ++w) {
			uchar* ptr = MatDST.ptr<uchar>(h, w);
			ptr[0] = (dst_up[2 * h][2 * w].b + dst_up[2 * h][2 * w + 1].b + dst_up[2 * h + 1][2 * w].b + dst_up[2 * h + 1][2 * w + 1].b) / 4;
			ptr[1] = (dst_up[2 * h][2 * w].g + dst_up[2 * h][2 * w + 1].g + dst_up[2 * h + 1][2 * w].g + dst_up[2 * h + 1][2 * w + 1].g) / 4;
			ptr[2] = (dst_up[2 * h][2 * w].r + dst_up[2 * h][2 * w + 1].r + dst_up[2 * h + 1][2 * w].r + dst_up[2 * h + 1][2 * w + 1].r) / 4;
			if ((dst_up[2 * h][2 * w].b + dst_up[2 * h][2 * w].g + dst_up[2 * h][2 * w].r) == 0 || (dst_up[2 * h][2 * w + 1].b + dst_up[2 * h][2 * w + 1].g + dst_up[2 * h][2 * w + 1].r) == 0 || (dst_up[2 * h + 1][2 * w].b + dst_up[2 * h + 1][2 * w].g + dst_up[2 * h + 1][2 * w].r) == 0 || (dst_up[2 * h + 1][2 * w + 1].b + dst_up[2 * h + 1][2 * w + 1].g + dst_up[2 * h + 1][2 * w + 1].r) == 0) {
				ptr[0] = 255;
				ptr[1] = 255;
				ptr[2] = 255;
			}
			if (dst_up[2 * h][2 * w].human % 2 || dst_up[2 * h][2 * w + 1].human % 2 || dst_up[2 * h + 1][2 * w].human % 2 || dst_up[2 * h + 1][2 * w + 1].human % 2)
				mask_fake[h][w] = 1;
			else
				mask_fake[h][w] = 0;
		}
	}
}


double Compute_PSNR(Mat& MatGT, Mat& MatDST) {
	int mask_cnt = 0;
	for (int h = 0; h < H_GT; ++h)
	{
		for (int w = 0; w < W_GT; ++w)
		{
			if (mask_real[h][w] == 1 || mask_fake[h][w] == 1) {
				mask_cnt++;
				uchar* ptr_real = MatGT.ptr<uchar>(h, w);
				uchar* ptr_fake = MatDST.ptr<uchar>(h, w);
				ptr_fake[0] = 1;
				ptr_real[0] = 1;
				ptr_fake[1] = 1;
				ptr_real[1] = 1;
				ptr_fake[2] = 1;
				ptr_real[2] = 1;
			}
		}
	}
	return getPSNR(MatGT, MatDST, mask_cnt);
}

int create_Directory(std::string path)
{
	int len = path.length();
	char tmpDirPath[256] = { 0 };
	for (int i = 0; i < len; i++)
	{
		tmpDirPath[i] = path[i];
		if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/')
		{
			if (_access(tmpDirPath, 0) == -1)
			{
				int ret = _mkdir(tmpDirPath);
				if (ret == -1) return ret;
			}
		}
	}
	return 0;
}

void write_img(Mat& MatGT, Mat& MatDST, int ref_frame, int dst_frame, int out_mode) {
	if (out_mode == 0) {
		create_Directory(outpath + to_string(ref_frame) + "\\");
		string fake_out = outpath + to_string(ref_frame) + "\\" + to_string(dst_frame) + "_dibr.jpg";
		imwrite(fake_out.data(), MatDST);
		string real_out = outpath + to_string(ref_frame) + "\\" + to_string(dst_frame) + "_gt.jpg";
		imwrite(real_out.data(), MatGT);
	}
	else if (out_mode == 1) {
		create_Directory(outpath + to_string(ref_frame) + "\\");
		string fake_out = outpath + to_string(ref_frame) + "\\" + to_string(dst_frame) + "_dibr_cache.jpg";
		imwrite(fake_out.data(), MatDST);
		string real_out = outpath + to_string(ref_frame) + "\\" + to_string(dst_frame) + "_gt.jpg";
		imwrite(real_out.data(), MatGT);
	}
	else if (out_mode == 2) {
		create_Directory(outpath + to_string(ref_frame) + "\\");
		string fake_out = outpath + to_string(ref_frame) + "\\" + to_string(dst_frame) + "_dibr.jpg";
		imwrite(fake_out.data(), MatDST);

	}
	else if (out_mode == 3) {
		create_Directory(outpath + to_string(ref_frame) + "\\");
		string fake_out = outpath + to_string(ref_frame) + "\\" + to_string(dst_frame) + "_dibr_cache.jpg";
		imwrite(fake_out.data(), MatDST);
	}
}

Mat Prepare_Ground_Truth(int gt_frame) {
	stringstream ss;
	ss << setw(5) << setfill('0') << gt_frame;
	string mask_path = inpath + ss.str() + "_id.png";
	getMask(mask_path, mask_real);

	string real_path = inpath + ss.str() + ".jpg";
	Mat MatGT = imread(real_path.data(), 1);
	return MatGT;
}

void Cache_Write(int frame_id, Location_3D location[H][W], int downsampling_ratio) {
	double threshold = 0.21 / 1.5;
	int block_cnt = 0;
	int block_size = downsampling_ratio;
	double dist;
	bool have_human = false;
	for (int h = 0; h * downsampling_ratio < H_GT; ++h)
		for (int w = 0; w * downsampling_ratio < W_GT; ++w) {
			int x = w * downsampling_ratio + downsampling_ratio / 2;
			int y = h * downsampling_ratio + downsampling_ratio / 2;

			have_human = false;
			for (int th = h * downsampling_ratio; th < (h + 1) * downsampling_ratio && !have_human; th++)
				for (int tw = w * downsampling_ratio; tw < (w + 1) * downsampling_ratio && !have_human; tw++)
					if (th >= 0 && tw >= 0 && th < H_GT && tw < W_GT && src[th][tw].human != 0) {
						have_human = true;
						break;
					}

			int x1 = w * downsampling_ratio;
			int x2 = (w + 1) * downsampling_ratio;
			int y1 = h * downsampling_ratio;
			int y2 = (h + 1) * downsampling_ratio;
			if (x1 < 0) x1 = 0;
			if (y1 < 0) y1 = 0;
			if (x2 > W_GT) x2 = W_GT;
			if (y2 > H_GT) y2 = H_GT;

			Location_3D location_down;
			location_down.x = (location[2 * y][2 * x].x + location[2 * y][2 * x + 1].x + location[2 * y + 1][2 * x].x + location[2 * y + 1][2 * x + 1].x) / 4;
			location_down.y = (location[2 * y][2 * x].y + location[2 * y][2 * x + 1].y + location[2 * y + 1][2 * x].y + location[2 * y + 1][2 * x + 1].y) / 4;
			location_down.z = (location[2 * y][2 * x].z + location[2 * y][2 * x + 1].z + location[2 * y + 1][2 * x].z + location[2 * y + 1][2 * x + 1].z) / 4;

			dist = tree.min_dist_from_location(location_down);
			if (dist > threshold) {
				KD_tree::_value block_inf;
				block_inf.ID = frame_id;
				block_inf.x = x;
				block_inf.y = y;
				block_inf.x1 = x1;
				if (!have_human) {
					block_inf.y1 = y1;
					block_inf.x2 = x2;
					block_inf.y2 = y2;
					block_inf.citations = 0;
					block_inf.last_frame = -1;
					block_inf.hfov = HFOV[frame_id];
					block_inf.wfov = WFOV[frame_id];
					block_inf.view_point = view_point[frame_id];
					pixel** block = new pixel *[block_size];
					for (int k = 0; k < block_size; k++)
						block[k] = new pixel[block_size];
					for (int i = y1; i < y2; i++)
						for (int j = x1; j < x2; j++)
							block[i - y1][j - x1] = src[i][j];
					block_inf.frame = block;
					tree.add_block_to_KD_tree(&location_down, &block_inf);
					block_cnt++;
				}
			}
		}
	cout << "#cached blocks: " << tree.current_elements << endl;
	output_log << "#cached blocks: " << tree.current_elements << endl;
}

void super_solution(pixel** f, int start_w, int end_w, int start_h, int end_h) {
	double step_i, step_j;
	int next_i, next_j;
	int cur_i, cur_j;
	for (int i = 0; i < end_h - start_h; ++i) {
		for (int j = 0; j < end_w - start_w; ++j) {

			cur_i = i;
			cur_j = j;
			if (cur_i + 1 < end_h - start_h)
				step_i = (f[cur_i + 1][cur_j].depth - f[cur_i][cur_j].depth);
			else
				step_i = 0;

			if (cur_j + 1 < end_w - start_w)
				step_j = (f[cur_i][cur_j + 1].depth - f[cur_i][cur_j].depth);
			else
				step_j = 0;

			if (step_i > 0.5 || step_i < -0.5) step_i = 0;
			if (step_j > 0.5 || step_j < -0.5) step_j = 0;
			step_i = step_i / up_size;
			step_j = step_j / up_size;
			for (int up_i = 0; up_i < up_size; up_i++)
				for (int up_j = 0; up_j < up_size; up_j++) {
					next_i = i * up_size + up_i;
					next_j = j * up_size + up_j;
					if (next_i < H && next_j < W) {
						src_block_up[next_i][next_j].b = f[cur_i][cur_j].b;
						src_block_up[next_i][next_j].g = f[cur_i][cur_j].g;
						src_block_up[next_i][next_j].r = f[cur_i][cur_j].r;
						src_block_up[next_i][next_j].depth = f[cur_i][cur_j].depth + step_i * up_i + step_j * up_j;
					}
				}
		}
	}
}

void set_corner(pixel** f, int start_w, int end_w, int start_h, int end_h) {
	int corner[4][2] = { { 0, 0 } ,{ 0, end_w - start_w - 1 } ,{ end_h - start_h - 1, end_w - start_w - 1 },{ end_h - start_h - 1, 0 } };
	int i, j;
	for (int k = 0; k < 4; ++k) {
		i = corner[k][0], j = corner[k][1];
		src_block_up[i][j].b = f[i][j].b;
		src_block_up[i][j].g = f[i][j].g;
		src_block_up[i][j].r = f[i][j].r;
		src_block_up[i][j].depth = f[i][j].depth;
	}
}

bool Cache_Read(Location_3D p, int dst_frame, int& frame_id, int& start_w, int& end_w, int& start_h, int& end_h, KD_tree::_value& tmp, int downsampling_ratio, int edge_h, int edge_w) {
	clock_t start = clock();
	double threshold = 0.21;
	double dist = tree.min_dist_from_location(p);
	if (dist > threshold) {
		return false;
	}
	int chosen_i = 0;
	KD_tree::node* node = tree.frame_lookup_by_location(p);
	KD_tree::ele** e = node->e;
	double min_dist = 100000;
	pixel** f;
	for (int i = 0; i < tree.split_threshold; i++) {
		if (!e[i]) {
			continue;
		}
		up_size = 1;
		f = e[i]->value.frame;
		start_w = e[i]->value.x1;
		end_w = e[i]->value.x2;
		start_h = e[i]->value.y1;
		end_h = e[i]->value.y2;
		set_corner(f, start_w, end_w, start_h, end_h);
		int corner_2D[8];
		inst.block_corner_estimation(W_GT, H_GT, W, H, start_w, end_w, start_h, end_h, e[i]->value.wfov, e[i]->value.hfov, WFOV[dst_frame], HFOV[dst_frame], src_block_up, e[i]->value.view_point, view_point[dst_frame], 0.15, corner_2D);

		int corner[4][2] = { { corner_2D[0], corner_2D[4] } ,{ corner_2D[1], corner_2D[5] },{ corner_2D[2], corner_2D[6] },{ corner_2D[3], corner_2D[7] } };
		int edge_p[2] = { edge_h, edge_w };
		if (IsPointInsideShape(edge_p, corner) == false) {
			continue;
		}
		dist = tree.point2point_distance(p, e[i]->key);
		if (dist > threshold) continue;
		if (dist < min_dist) {
			min_dist = dist;
			tmp = e[i]->value;
			chosen_i = i;
		}
	}
	if (min_dist == 100000) {
		return false;
	}

	frame_id = tmp.ID;
	start_w = tmp.x1;
	end_w = tmp.x2;
	start_h = tmp.y1;
	end_h = tmp.y2;
	
	if (e[chosen_i]->value.last_frame == cur_frame_identity) {
		return false;
	}
	else {
		e[chosen_i]->value.last_frame = cur_frame_identity;
	}
	chosen_e = e[chosen_i];

	f = tmp.frame;
	up_size = 1;
	set_corner(f, start_w, end_w, start_h, end_h);
	double scale = inst.block_super_resolution_estimation(W_GT, H_GT, W, H, up_size * start_w, up_size * end_w, up_size * start_h, up_size * end_h, tmp.wfov, tmp.hfov, WFOV[dst_frame], HFOV[dst_frame], src_block_up, tmp.view_point, view_point[dst_frame], 0.15);
	up_size = floor(scale); up_size = 1;
	if (up_size <= 1)
		up_size = 1;
	else {
		if (up_size > 16) up_size = 16;
	}
	inpainting_size = min(20, max(1, int(scale)));
	super_solution(f, start_w, end_w, start_h, end_h);
	return true;
}


int DIBR(int ref_frame, int dst_frame, int mask_mode, int out_mode) {
	for (int h = 0; h < H; ++h)
		for (int w = 0; w < W; ++w)
			dst_up[h][w].clear();

	inst.perspective_transform(W, H, 0, W, 0, H, WFOV[ref_frame], HFOV[ref_frame], WFOV[dst_frame], HFOV[dst_frame], src_up, dst_up, view_point[ref_frame], view_point[dst_frame], 0.15, 0, src_location, dst_location);

	inst.inpainting(0, W, 0, H, dst_up);

	Mat MatDST(1080, 1920, CV_8UC3);
	get_DST(MatDST);
	Mat MatGT = Prepare_Ground_Truth(dst_frame);
	double psnr = Compute_PSNR(MatGT, MatDST);

	write_img(MatGT, MatDST, ref_frame, dst_frame, out_mode);
	cout << " psnr(dibr): " << psnr << ';';
	output_log << " psnr(dibr): " << psnr << ';';

	return 0;
}


bool whether_crack(int h, int w, int inpainting_dis) {
	inpainting_dis = inpainting_dis / 2;
	for (int i = -inpainting_dis; i <= inpainting_dis; i++)
		for (int j = -inpainting_dis; j <= inpainting_dis; j++)
			if (abs(i) + abs(j) <= inpainting_dis)
				if (dst_up[h + i][w + j].valid == 1)
					return true;
	return false;
}

void write_img_entire(string path) {
	Mat show(H, W, CV_8UC3);
	for (int h = 0; h < H; h++)
		for (int w = 0; w < W; w++) {
			uchar* ptr = show.ptr<uchar>(h, w);
			ptr[0] = dst_up[h][w].b;
			ptr[1] = dst_up[h][w].g;
			ptr[2] = dst_up[h][w].r;
		}
	imwrite(path.data(), show);
}

void fill_hole(Location_3D p, int dst_frame, int downsampling_ratio, int last_h, int last_w) {
	double a = view_point[dst_frame].x, b = view_point[dst_frame].y, c = view_point[dst_frame].z;
	int frame_id, start_w, end_w, start_h, end_h;
	int num = 3;
	Location_3D tp;
	KD_tree::_value tmp;
	bool have_near = false;
	for (int i = num - 1; i >= 0 && !have_near; i--) {
		tp.x = ((p.x * (num - i)) + a * i) / (num * 1.0);
		tp.y = ((p.y * (num - i)) + b * i) / (num * 1.0);
		tp.z = ((p.z * (num - i)) + c * i) / (num * 1.0);
		if (Cache_Read(tp, dst_frame, frame_id, start_w, end_w, start_h, end_h, tmp, downsampling_ratio, last_h, last_w)) {
			have_near = true;
			memset(changed, false, sizeof(changed));
			double warping_PSNR, no_warping_PSNR;
			int change_cnt = inst.perspective_transform_block(up_size * W_GT, up_size * H_GT, W, H, up_size * start_w, up_size * end_w, up_size * start_h, up_size * end_h, tmp.wfov, tmp.hfov, WFOV[dst_frame], HFOV[dst_frame], src_block_up, dst_up, tmp.view_point, view_point[dst_frame], 0.15, 0, dst_location, changed, change_h, change_w, MatGT, mask_real, dst_inpainting, warping_PSNR, no_warping_PSNR);
			if (change_cnt > 0) {
				int corner_2D[10];
				inst.block_corner_estimation(W_GT, H_GT, W, H, start_w, end_w, start_h, end_h, tmp.wfov, tmp.hfov, WFOV[dst_frame], HFOV[dst_frame], src_block_up, tmp.view_point, view_point[dst_frame], 0.15, corner_2D);
				int corner[4][2] = { { corner_2D[0], corner_2D[4] } ,{ corner_2D[1], corner_2D[5] },{ corner_2D[2], corner_2D[6] },{ corner_2D[3], corner_2D[7] } };
				inst.inpainting(change_cnt, change_h, change_w, dst_up, corner, inpainting_size, dst_location);
			}
			if (warping_PSNR > 0.001) {
				chosen_e->value.update_performance_gain(warping_PSNR, no_warping_PSNR);
			}
		}
	}
}


void Scan(int dst_frame, int downsampling_ratio, int inpainting_dis) {
	bool tag;
	int query_cnt = 0;
	int last_h = 0, last_w = 0, dis_last = 0;
	for (int h = H - 1; h >= 0; --h) {
		tag = false;
		for (int w = W - 1; w >= 0; --w) {
			if (tag && dst_up[h][w].valid == 0 && !used_p[last_h][last_w] && !used_p[h][w] && !whether_crack(h, w, inpainting_dis)) {
				used_p[last_h][last_w] = true;
				used_p[h][w] = true;
				query_cnt++;
				fill_hole(dst_location[last_h][last_w], dst_frame, downsampling_ratio, last_h, last_w);
				tag = false;
			}
			if (dst_up[h][w].valid == 1 && dst_up[h][w].human != 1) {
				last_h = h;
				last_w = w;
				tag = true;
			}
		}
	}

	for (int w = 0; w < W; ++w) {
		tag = false;
		for (int h = H - 1; h >= 0; --h) {
			if (tag && dst_up[h][w].valid == 0 && !used_p[last_h][last_w] && !used_p[h][w] && !whether_crack(h, w, inpainting_dis)) {
				query_cnt++;
				used_p[last_h][last_w] = true;
				used_p[h][w] = true;
				fill_hole(dst_location[last_h][last_w], dst_frame, downsampling_ratio, last_h, last_w);
				tag = false;
			}
			if (dst_up[h][w].valid == 1 && dst_up[h][w].human != 1) {
				last_h = h;
				last_w = w;
				tag = true;
			}
		}
	}

	for (int h = 0; h < H; ++h) {
		tag = false;
		for (int w = 0; w < W; ++w) {
			if (tag && dst_up[h][w].valid == 0 && !used_p[last_h][last_w] && !used_p[h][w] && !whether_crack(h, w, inpainting_dis)) {
				query_cnt++;
				used_p[last_h][last_w] = true;
				used_p[h][w] = true;
				fill_hole(dst_location[last_h][last_w], dst_frame, downsampling_ratio, last_h, last_w);
				tag = false;
			}
			if (dst_up[h][w].valid == 1 && dst_up[h][w].human != 1) {
				last_h = h;
				last_w = w;
				tag = true;
			}
		}
	}


	for (int w = W - 1; w >= 0; --w) {
		tag = false;
		for (int h = 0; h < H; ++h) {
			if (tag && dst_up[h][w].valid == 0 && !used_p[last_h][last_w] && !used_p[h][w] && !whether_crack(h, w, inpainting_dis)) {
				query_cnt++;
				used_p[last_h][last_w] = true;
				used_p[h][w] = true;
				fill_hole(dst_location[last_h][last_w], dst_frame, downsampling_ratio, last_h, last_w);
				tag = false;
			}
			if (dst_up[h][w].valid == 1 && dst_up[h][w].human != 1) {
				last_h = h;
				last_w = w;
				tag = true;
			}
		}
	}
}


int DIBR_multi(int ref_frame, int dst_frame, int mask_mode, int out_mode) {
	for (int h = 0; h < H; ++h)
		for (int w = 0; w < W; ++w)
			dst_up[h][w].clear();

	inst.perspective_transform(W, H, 0, W, 0, H, WFOV[ref_frame], HFOV[ref_frame], WFOV[dst_frame], HFOV[dst_frame], src_up, dst_up, view_point[ref_frame], view_point[dst_frame], 0.15, 0, src_location, dst_location);
	copy_mat(dst_up, dst_inpainting);
	inst.inpainting(0, W, 0, H, dst_inpainting);
	MatGT = Prepare_Ground_Truth(dst_frame);

	memset(used_p, false, sizeof(used_p));

	inst.inpainting(0, W, 0, H, dst_up, 2);
	Scan(dst_frame, downsampling_ratio, inpainting_dis);

	inst.inpainting(0, W, 0, H, dst_up);
	Mat MatDST(1080, 1920, CV_8UC3);
	get_DST(MatDST);

	double psnr = Compute_PSNR(MatGT, MatDST);
	write_img(MatGT, MatDST, ref_frame, dst_frame, out_mode);
	cout << " psnr(dibr+cache): " << psnr << ';';
	output_log << " psnr(dibr+cache): " << psnr << ';';

	if (dst_frame - ref_frame == FRAME_CNT) {
		cout << endl;
		output_log << endl;
		Cache_Write(ref_frame, src_location, downsampling_ratio);
	}

	return 0;
}


int main() {
	tree.init(3, 30, 15, 40000);
	outpath = abs_dir + "results\\";
	output_log.open(outpath + "log.txt");

	abs_dir = ".\\";
	seq_name = "2020-06-03-22-25-09";
	inpath = abs_dir + "data\\" + seq_name + "\\";
	human_id = 101;
	cam_near_clip = 0.15;
	cam_far_clip = 8000;
	read_viewpoint_euler(inpath + "viewpoints.txt");
	for (int i = 0; i < 500; i += 20) {
		prepare(i, 1, 1);
		inst.perspective_transform(W, H, 0, W, 0, H, WFOV[i], HFOV[i], WFOV[i + 1], HFOV[i + 1], src_up, dst_up, view_point[i], view_point[i + 1], 0.15, -1, src_location, dst_location);
		Cache_Write(i, src_location, downsampling_ratio);
	}

	seq_name = "2020-06-03-20-28-01";
	inpath = abs_dir + "data\\" + seq_name + "\\";
	human_id = 62;
	frame_number = 1165;
	read_viewpoint_euler(inpath + "viewpoints.txt");
	for (int i = 700; i < frame_number - FRAME_CNT; i += 10) {
		prepare(i, 1, 1);
		cout << "input frame: " << i;
		output_log << "input frame: " << i;
		for (int t = 1; t <= FRAME_CNT; t++) {
			cur_frame_identity++;
			cout << endl << "   predict frame: " << i + t << ';';
			output_log << endl << "   predict frame: " << i + t << ';';
			DIBR(i, i + t, 0, 2);
			DIBR_multi(i, i + t, 0, 1);
		}
		tree.delete_last_negative_block();
	}
	output_log.close();
}