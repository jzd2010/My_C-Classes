#include "Read_Info_From_File.h"
#include "Include_OpenCVHeadFile.h"

Read_Info_From_File::Read_Info_From_File(void)
{
}


Read_Info_From_File::~Read_Info_From_File(void)
{
}

void Read_Info_From_File::read_Face_Eye_Info(string &filePath,vector<string> &imageFile,vector<FacePos> &facerects,vector<string> &lables,vector<EyePos> &eyespos)
{
	char separator =';';
	std::ifstream file(filePath.c_str(), ifstream::in);
	if (!file) {
		//::MessageBox(NULL,"无法打开该文件路径","Error",MB_OK);
		return;
	}
	EyePos eyepos;
	FacePos facerect;
	string line, path, rectX,rectY,rectW,rectH,leftx,lefty,rightx,righty,lable;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);//分离路径		
		getline(liness,rectX, separator);//分离X坐标
		getline(liness,rectY, separator);//分离Y坐标
		getline(liness,rectW, separator);//分离宽度
		getline(liness,rectH, separator);//分离高度
		getline(liness,lable,separator);
		getline(liness,leftx, separator);//分离X坐标
		getline(liness,lefty, separator);//分离Y坐标
		getline(liness,rightx, separator);//分离宽度
		getline(liness,righty, separator);//分离高度
		if(!path.empty() && !rectX.empty() && !rectY.empty() &&! rectW.empty() && !rectH.empty()
			&&!leftx.empty()&&!lefty.empty()&&!rightx.empty()&&!righty.empty()&&!lable.empty())
		{
			lables.push_back(lable);
			facerect.left = atoi(rectX.c_str());facerect.top = atoi(rectY.c_str());
			facerect.width = atoi(rectW.c_str());facerect.height = atoi(rectH.c_str());
			facerects.push_back(facerect);
			imageFile.push_back(path);
			eyepos.leftx = atoi(leftx.c_str());eyepos.lefty = atoi(lefty.c_str());
			eyepos.rightx = atoi(rightx.c_str());eyepos.righty = atoi(righty.c_str());
			eyespos.push_back(eyepos);
		}
	}
	file.close();
	return ;
}

void Read_Info_From_File::read_Face_Info(string &filePath,vector<string> &imageFile,vector<FacePos> &facerects,vector<string> &lables)
{
	char separator =';';
	std::ifstream file(filePath.c_str(), ifstream::in);
	if (!file) {
		//::MessageBox(NULL,"无法打开该文件路径","Error",MB_OK);
		return;
	}
	EyePos eyepos;
	FacePos facerect;
	string line, path, rectX,rectY,rectW,rectH,lable;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);//分离路径		
		getline(liness,rectX, separator);//分离X坐标
		getline(liness,rectY, separator);//分离Y坐标
		getline(liness,rectW, separator);//分离宽度
		getline(liness,rectH, separator);//分离高度
		getline(liness,lable,separator);
		if(!path.empty() && !rectX.empty() && !rectY.empty() &&! rectW.empty() && !rectH.empty()&&!lable.empty())
		{
			lables.push_back(lable);
			facerect.left = atoi(rectX.c_str());facerect.top = atoi(rectY.c_str());
			facerect.width = atoi(rectW.c_str());facerect.height = atoi(rectH.c_str());
			facerects.push_back(facerect);
			imageFile.push_back(path);
		}
	}
	file.close();
	return ;
}


void Read_Info_From_File::read_Eye_Info(string &filePath,vector<string> &imageFile,vector<EyePos> &eyespos,vector<string> &lables)
{
	char separator =';';
	std::ifstream file(filePath.c_str(), ifstream::in);
	if (!file) {
		//::MessageBox(NULL,"无法打开该文件路径","Error",MB_OK);
		return;
	}
	EyePos eyepos;
	FacePos facerect;
	string line, path, rectX,rectY,rectW,rectH,leftx,lefty,rightx,righty,lable;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);//分离路径	
		getline(liness,leftx, separator);//分离X坐标
		getline(liness,lefty, separator);//分离Y坐标
		getline(liness,rightx, separator);//分离宽度
		getline(liness,righty, separator);//分离高度
		getline(liness,lable,separator);
		if(!path.empty() &&!leftx.empty()&&!lefty.empty()&&!rightx.empty()&&!righty.empty()&&!lable.empty())
		{
			lables.push_back(lable);
			imageFile.push_back(path);
			eyepos.leftx = atoi(leftx.c_str());eyepos.lefty = atoi(lefty.c_str());
			eyepos.rightx = atoi(rightx.c_str());eyepos.righty = atoi(righty.c_str());
			eyespos.push_back(eyepos);
		}
	}
	file.close();
	return ;
}

