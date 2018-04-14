#pragma once
class Cal_LBP_Hist
{
public:
	static Mat spatial_histogram(InputArray _src, int numPatterns,
		int grid_x, int grid_y, bool normed);
	Cal_LBP_Hist(void);
	~Cal_LBP_Hist(void);
};

