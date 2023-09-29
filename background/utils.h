#pragma once
#include<algorithm>
using namespace std;

inline bool IsPointInsideShape(int p[2], int corner[4][2])
{
    int nCross = 0;
    int nCount = 4;
    int p_x = p[0];
    int p_y = p[1];
    for (int i = 0; i < nCount; i++)
    {
        int p1_x = corner[i][0];
        int p1_y = corner[i][1];
        int p2_x = corner[(i + 1) % nCount][0];
        int p2_y = corner[(i + 1) % nCount][1];
        // 求解 y=p.y 与 p1p2 的交点
        if (p1_y == p2_y)
        {// p1p2 与 y=p0.y平行
            continue;
        }
        if (p_y < min(p1_y, p2_y))
        {// 交点在p1p2延长线上
            continue;
        }
        if (p_y >= max(p1_y, p2_y))
        {// 交点在p1p2延长线上
            continue;
        }
        // 求交点的 X 坐标 --------------------------------------------------------------
        double x = (double)(p_y - p1_y) * (double)(p2_x - p1_x) / (double)(p2_y - p1_y) + p1_x;
        if (x > p_x) {
            nCross++; // 只统计单边交点
        }
    }
    // 单边交点为偶数，点在多边形之外
//    if (nCross % 2 == 1) {
//        log("在多边形内");
//    }
//    if (nCross % 2 == 0) {
//        log("在多边形外");
//    }
    return (nCross % 2 == 1);
}