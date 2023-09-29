#pragma once

#include <iostream>
#include <cstdio>
#include <math.h> 
#include <algorithm>
#include "world.hpp"

using namespace std;

class KD_tree
{
public:
	/*
	dimensions: the dimension of the space.
	split_threshold: split the node into 2 child nodes when it contains >= split_threshold 3D locations.
	split_threshold: merge the 2 child nodes into one node when they contain <= merge_threshold 3D locations in total.
	max_ele_number: the maximum number of elements in the KD tree.
	current_elements: the number of elements currently.
	*/
	int dimensions, split_threshold, merge_threshold, max_ele_number, current_elements;
	double evict_warping_gain;

	int current_time;

	struct _value
	{
		//ID: the frame ID
		//(x,y) the coordinate of the pixel in the frame.
		//the number of citations
		int ID, x, y, citations;
		//the frame matrix
		pixel** frame;
		//(x1,y1) indicating the top-left corner, and (x2, y2) indicating the bottom-right corner
		int x1, x2, y1, y2;
		double hfov, wfov;
		viewpoint view_point;
		//last_frame indicates in which frame the block was last used.
		int last_frame;

		//sum_of_warping_PSNR indicates the sum of PSNR after image warping of the corresponding area when this block is used.
		//sum_of_no_warping_PSNR indicates the sum of PSNR of the corresponding area when this block is not used.
		double sum_of_warping_PSNR, sum_of_no_warping_PSNR;
		double last_warping_PSNR, last_no_warping_PSNR;

		/*
		update the PSNR before/after warping when a block is used.
		warping_PSNR: PSNR after warping.
		no_warping_PSNR: PSNR without warping.
		*/
		void update_performance_gain(double warping_PSNR, double no_warping_PSNR)
		{
			citations++;
			sum_of_warping_PSNR += warping_PSNR;
			sum_of_no_warping_PSNR += no_warping_PSNR;

			last_warping_PSNR = warping_PSNR;
			last_no_warping_PSNR = no_warping_PSNR;
		}
	};

	//An entry in the dictionary.
	struct ele
	{
		Location_3D key;
		_value value;
		int timestamp;
	};

	struct node
	{
		node* left, * right;//the pointer to the left child and the right child. =0 if no child
		bool leaf;//whether the node is a leaf node (leaf == 1) or not (leaf == 0)
		ele** e;// an array of a set of ele*. Each ele* represents a 3D location-_value pair.
		//items: the number of valid ele* in e.
		//split_dimension: 0 indicating that this node splits eles by x axis, 
		//1 indicating that this node splits eles by y axis.
		//2 indicating that this node splits eles by z axis.
		//-1 indicating that this node is not splited yet.
		int items, split_dimension;
		//all eles in this node are within the range of ([x_low, x_high], [y_low, y_high]).
		double x_low, x_high, y_low, y_high, z_low, z_high;

		/*
		node initialization.
		split_threshold: split the node when it contains more than split_threshold elements.
		x_low, x_high , y_low , y_high: the range of this node.
		dimension: in which dimension this node split elements into 2 child nodes.
		*/
		node(int split_threshold, double xlow, double xhigh, double ylow, double yhigh, double zlow, double zhigh)
		{
			left = right = 0;
			leaf = 1;
			items = 0;
			e = (ele**)malloc(sizeof(ele*) * split_threshold);
			for (int i = 0; i < split_threshold; ++i)
				e[i] = 0;
			x_low = xlow, x_high = xhigh, y_low = ylow, y_high = yhigh, z_low = zlow, z_high = zhigh;
			split_dimension = -1;
		}
	};

	/*
	point to point eucilid distance.
	a: the 3D location.
	b: another 3D location.
	*/
	double point2point_distance(Location_3D a, Location_3D b)
	{
		double x = a.x - b.x, y = a.y - b.y, z = a.z - b.z;
		return sqrt(x * x + y * y + z * z);
	}

private:
	node* root;

	/*
	compare elements by x axis
	*/
	static bool cmpx(ele* a, ele* b)
	{
		return a->key.x < b->key.x;
	}

	/*
	comapre elements by y axis.
	*/
	static bool cmpy(ele* a, ele* b)
	{
		return a->key.y < b->key.y;
	}

	/*
	comapre elements by z axis.
	*/
	static bool cmpz(ele* a, ele* b)
	{
		return a->key.z < b->key.z;
	}

	/*
	split node a into 2 child nodes.
	a: the node to be split.
	*/
	void split(node* a)
	{
		node* l, * r;
		double mid;
		double minx = 100000, maxx = -100000, miny = 100000, maxy = -100000, minz = 100000, maxz = -100000;
		for (int i = 0; i < split_threshold; ++i) {
			minx = min(minx, a->e[i]->key.x);
			maxx = max(maxx, a->e[i]->key.x);
			miny = min(miny, a->e[i]->key.y);
			maxy = max(maxy, a->e[i]->key.y);
			minz = min(minz, a->e[i]->key.z);
			maxz = max(maxz, a->e[i]->key.z);
		}
		double delta_x = maxx - minx, delta_y = maxy - miny, delta_z = maxz - minz;
		double max_delta = max(delta_x, max(delta_y, delta_z));
		if (max_delta == delta_x) a->split_dimension = 0;
		if (max_delta == delta_y) a->split_dimension = 1;
		if (max_delta == delta_z) a->split_dimension = 2;

		if (a->split_dimension == 0) {
			//split the node by x dimension
			sort(a->e, a->e + split_threshold, cmpx);
			mid = (a->e[split_threshold >> 1]->key.x + a->e[(split_threshold >> 1) - 1]->key.x) / 2;
			l = new node(split_threshold, a->x_low, mid, a->y_low, a->y_high, a->z_low, a->z_high);
			r = new node(split_threshold, mid, a->x_high, a->y_low, a->y_high, a->z_low, a->z_high);
		}
		else if (a->split_dimension == 1) {
			//split the node by y dimension
			sort(a->e, a->e + split_threshold, cmpy);
			mid = (a->e[split_threshold >> 1]->key.y + a->e[(split_threshold >> 1) - 1]->key.y) / 2;
			l = new node(split_threshold, a->x_low, a->x_high, a->y_low, mid, a->z_low, a->z_high);
			r = new node(split_threshold, a->x_low, a->x_high, mid, a->y_high, a->z_low, a->z_high);
		}
		else {
			//split the node by z dimension
			sort(a->e, a->e + split_threshold, cmpz);
			mid = (a->e[split_threshold >> 1]->key.z + a->e[(split_threshold >> 1) - 1]->key.z) / 2;
			l = new node(split_threshold, a->x_low, a->x_high, a->y_low, a->y_high, a->z_low, mid);
			r = new node(split_threshold, a->x_low, a->x_high, a->y_low, a->y_high, mid, a->z_high);
		}

		l->items = split_threshold >> 1;
		for (int i = 0; i < split_threshold >> 1; ++i) {
			l->e[i] = a->e[i];
		}

		r->items = split_threshold >> 1;
		for (int i = 0; i < split_threshold >> 1; ++i) {
			r->e[i] = a->e[i + (split_threshold >> 1)];
		}

		a->items = 0;
		for (int i = 0; i < split_threshold; ++i)
			a->e[i] = 0;
		a->leaf = 0;
		a->left = l;
		a->right = r;
	}

	/*
	merge two child nodes into one node, if they satisfy the merge condition.
	a: the father node of the two nodes to be merged.
	*/
	void merge(node* a)
	{
		if (!a) return;
		if (!a->left->leaf || !a->right->leaf) return;
		if (a->left->items + a->right->items >= merge_threshold) return;
		a->leaf = 1;
		a->items = 0;
		for (int i = 0; i < split_threshold; ++i)
			if (a->left->e[i]) a->e[a->items++] = a->left->e[i];
		for (int i = 0; i < split_threshold; ++i)
			if (a->right->e[i]) a->e[a->items++] = a->right->e[i];
		a->x_low = min(a->left->x_low, a->right->x_low);
		a->y_low = min(a->left->y_low, a->right->y_low);
		a->z_low = min(a->left->z_low, a->right->z_low);
		a->x_high = max(a->left->x_high, a->right->x_high);
		a->y_high = max(a->left->y_high, a->right->y_high);
		a->z_high = max(a->left->z_high, a->right->z_high);
		free(a->left);
		free(a->right);
		a->left = a->right = 0;
	}

	/*
	whether p is inside the range of a.
	*/
	bool inside(Location_3D* p, node* a)
	{
		return (p->x >= a->x_low && p->x <= a->x_high && p->y >= a->y_low && p->y <= a->y_high && p->z >= a->z_low && p->z <= a->z_high);
	}

	void add_point(node* a, Location_3D* p, _value* v)
	{
		if (a->leaf && a->items == split_threshold) split(a);
		if (a->leaf) {
			for (int i = 0; i < split_threshold; ++i)
				if (!a->e[i]) {
					a->e[i] = new ele();
					memcpy(&(a->e[i]->key), p, sizeof(Location_3D));
					memcpy(&(a->e[i]->value), v, sizeof(_value));
					a->e[i]->timestamp = current_time++;
					++a->items;
					++current_elements;
					break;
				}
		}
		else {
			if (inside(p, a->left)) add_point(a->left, p, v); else add_point(a->right, p, v);
		}
	}

	inline bool check_eviction(double no_warping_PSNR, double warping_PSNR, double current_time, double timestamp, double rate)
	{	
		return (warping_PSNR - no_warping_PSNR) / (current_time - timestamp) < rate;
	}

	inline bool check_eviction_LRU(int citation, double current_time, double timestamp, double rate)
	{	
		return (double)citation / (current_time - timestamp) < rate;
	}

	inline bool check_eviction_FIFO(double current_time, double timestamp, double rate)
	{
		return 1.0 / (current_time - timestamp) < rate;
	}

	void delete_block_from_tree(double rate, node* p)
	{
		if (!p) return;
		delete_block_from_tree(rate, p->left);
		delete_block_from_tree(rate, p->right);
		if (!p->leaf) merge(p); else {
			for (int i = 0; i < split_threshold; ++i)
				if (p->e[i] && check_eviction(p->e[i]->value.sum_of_no_warping_PSNR, p->e[i]->value.sum_of_warping_PSNR, current_time, p->e[i]->timestamp, rate)) {
				//if (p->e[i] && check_eviction_LRU(p->e[i]->value.citations, current_time, p->e[i]->timestamp, rate)) {
				//if (p->e[i] && check_eviction_FIFO(current_time, p->e[i]->timestamp, rate)) {
					for (int k = 0; k < 64; k++)
						delete[] p->e[i]->value.frame[k];
					delete[] p->e[i]->value.frame;
					free(p->e[i]);
					p->e[i] = 0;
					--p->items;
					--current_elements;
				}
		}
	}

	void delete_last_negative_block_from_tree(node* p)
	{
		if (!p) return;
		delete_last_negative_block_from_tree(p->left);
		delete_last_negative_block_from_tree(p->right);
		if (!p->leaf) merge(p); else {
			for (int i = 0; i < split_threshold; ++i)
				if (p->e[i] && (p->e[i]->value.last_no_warping_PSNR > p->e[i]->value.last_warping_PSNR)) {
					for (int k = 0; k < 64; k++)
						delete[] p->e[i]->value.frame[k];
					delete[] p->e[i]->value.frame;
					free(p->e[i]);
					p->e[i] = 0;
					--p->items;
					--current_elements;
				}
		}
	}

	node* frame_lookup(node* a, Location_3D p)
	{
		//printf("Lookup %.2lf %.2lf %.2lf\n", p.x, p.y, p.z);
		//printf("Node: %.2lf %.2lf %.2lf %.2lf\n", a->x_low, a->x_high, a->y_low, a->y_high);
		if (a->leaf) {
			return a;
		}
		else {
			if (inside(&p, a->left) && (!a->left->leaf || a->left->items)) return frame_lookup(a->left, p); else return frame_lookup(a->right, p);
		}
	}

	double min_dist(Location_3D p, node* a)
	{
		double ret = 100000;
		if (a->leaf) {
			for (int i = 0; i < split_threshold; ++i)
				if (a->e[i]) ret = min(ret, point2point_distance(p, a->e[i]->key));
			return ret;
		}
		else {
			if (inside(&p, a->left) && (!a->left->leaf || a->left->items)) return min_dist(p, a->left); else return min_dist(p, a->right);
		}
	}

	void print_tree(node* p)
	{
		if (!p) return;
		printf("Node: %.2lf %.2lf %.2lf %.2lf %.2lf %.2lf\n", p->x_low, p->x_high, p->y_low, p->y_high, p->z_low, p->z_high);
		for (int i = 0; i < split_threshold; ++i)
			if (p->e[i]) {
				printf("%.2lf %.2lf %.2lf\n", p->e[i]->key.x, p->e[i]->key.y, p->e[i]->key.z);
			}
		print_tree(p->left);
		print_tree(p->right);
	}

	void print_citation(node* p, FILE* f)
	{
		if (!p) return;
		for (int i = 0; i < split_threshold; ++i)
			if (p->e[i]) {
				fprintf(f, "%.2lf %.2lf %.2lf %d %.2lf %.2lf %d\n", p->e[i]->key.x, p->e[i]->key.y, p->e[i]->key.z, p->e[i]->value.citations, p->e[i]->value.sum_of_warping_PSNR, p->e[i]->value.sum_of_no_warping_PSNR, current_time - p->e[i]->timestamp);
			}
		print_citation(p->left, f);
		print_citation(p->right, f);
	}

	/*
	delete all elements in a frame.
	frame_ID: the frame ID to be deleted.
	*/
	void delete_block(double rate)
	{
		delete_block_from_tree(rate, root);
	}

public:
	/*
	delete the blocks which have negative influence on image warping PSNR.
	*/
	void delete_negative_block()
	{
		delete_block_from_tree(0, root);
	}

	/*
	delete the blocks which have negative influence on image warping PSNR.
	*/
	void delete_last_negative_block()
	{
		delete_last_negative_block_from_tree(root);
	}

	/*
	KD_tree initialization.
	d: dimensions.
	split: the maximum number of a node can contain.
	merge: merge two child nodes if they contains < min_ele elements in total.
	max_elements: the maximum number of elements in the KD tree.
	*/
	void init(int d, int split, int merge, int max_elements)
	{
		dimensions = d;
		split_threshold = split;
		merge_threshold = merge;
		max_ele_number = max_elements;
		current_elements = 0;
		root = new node(split_threshold, -100000, 100000, -100000, 100000, -100000, 100000);
		current_time = 0;
		evict_warping_gain = -0.0000001;//please adjust this parameter.
	}

	/*
	add a 3D location to the KD_tree.
	p: the 3D location to be added to the tree.
	block: the pointer to a constructed _value structure.
	*/
	void add_block_to_KD_tree(Location_3D* p, _value* block)
	{
		block->sum_of_no_warping_PSNR = block->sum_of_warping_PSNR = block->last_no_warping_PSNR = block->last_warping_PSNR = 0;
		if (current_elements >= max_ele_number) {
			while (1) {
				delete_block(evict_warping_gain);//cout << current_elements << endl;
				if (current_elements <= 3 * max_ele_number / 4) {
					if (current_elements < max_ele_number / 2)
						//lower the evict bar
						if (evict_warping_gain >= 0) evict_warping_gain = (evict_warping_gain - 0.00000005) * 0.8; else evict_warping_gain = (evict_warping_gain - 0.00000005) * 1.2;
					break;
				}
				//upper the evict bar
				if (evict_warping_gain >= 0) evict_warping_gain = (evict_warping_gain + 0.00000005) * 1.2; else evict_warping_gain = (evict_warping_gain + 0.00000005) * 0.8;
			}
		}
		add_point(root, p, block);
	}

	/*
	print the KD tree. Only useful in debuging.
	*/
	void print_KD_tree()
	{
		print_tree(root);
	}

	/*
	print the citation of blocks.
	*/
	void print_block_citation(FILE* fout, int frame_id)
	{
		fprintf(fout, "\nframe id: %d, current elements: %d\n", frame_id, current_elements);
		print_citation(root, fout);
	}

	/*
	Given a 3D location, return a node which contains a set of 3D locations in the dictionary which are close to the requested location.
	p: the requested 3D location.
	*/
	node* frame_lookup_by_location(Location_3D p)
	{
		++current_time;
		return frame_lookup(root, p);
	}

	/*
	given a 3D location, return a 3D location in the dictionary which is the closest one to the requested 3D location.
	p: the requested 3D location.
	*/
	double min_dist_from_location(Location_3D p)
	{
		return min_dist(p, root);
	}
};
