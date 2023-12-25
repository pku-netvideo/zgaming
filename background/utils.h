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
        // ��� y=p.y �� p1p2 �Ľ���
        if (p1_y == p2_y)
        {// p1p2 �� y=p0.yƽ��
            continue;
        }
        if (p_y < min(p1_y, p2_y))
        {// ������p1p2�ӳ�����
            continue;
        }
        if (p_y >= max(p1_y, p2_y))
        {// ������p1p2�ӳ�����
            continue;
        }
        // �󽻵�� X ���� --------------------------------------------------------------
        double x = (double)(p_y - p1_y) * (double)(p2_x - p1_x) / (double)(p2_y - p1_y) + p1_x;
        if (x > p_x) {
            nCross++; // ֻͳ�Ƶ��߽���
        }
    }
    // ���߽���Ϊż�������ڶ����֮��
//    if (nCross % 2 == 1) {
//        log("�ڶ������");
//    }
//    if (nCross % 2 == 0) {
//        log("�ڶ������");
//    }
    return (nCross % 2 == 1);
}