#pragma once
#include "Include_OpenCVHeadFile.h"

typedef struct FacePos
{
	short int left;
	short int top;
	short int width;
	short int height;
}FacePos;

typedef struct EyePos
{
	short int leftx;
	short int lefty;
	short int rightx;
	short int righty;
}EyePos;

class Read_Info_From_File
{
public:
	void read_Face_Eye_Info(string &filePath,vector<string> &imageFile,vector<FacePos> &facerects,vector<string> &lables,vector<EyePos> &eyespos);
	void read_Face_Info(string &filePath,vector<string> &imageFile,vector<FacePos> &facerects,vector<string> &lables);
	void read_Eye_Info(string &filePath,vector<string> &imageFile,vector<EyePos> &eyespos,vector<string> &lables);
	Read_Info_From_File(void);
	~Read_Info_From_File(void);
};

