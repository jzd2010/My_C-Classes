#include "stdafx.h"
#include "stdlib.h"
#include "direct.h"
#include "string.h"
#include "io.h"
#include "stdio.h" 
#include <string>
#include "CBrowseDir.h"


CBrowseDir::CBrowseDir()
{
	//用当前目录初始化m_szInitDir
	getcwd(m_szInitDir,_MAX_PATH);

	//如果目录的最后一个字母不是'\',则在最后加上一个'\'
	int len=strlen(m_szInitDir);
	if (m_szInitDir[len-1] != '\\')
		strcat(m_szInitDir,"\\");
}

bool CBrowseDir::SetInitDir(const char *dir)
{
	//先把dir转换为绝对路径
	if (_fullpath(m_szInitDir,dir,_MAX_PATH) == NULL)
		return false;

	//判断目录是否存在
	if (_chdir(m_szInitDir) != 0)
		return false;

	//如果目录的最后一个字母不是'\',则在最后加上一个'\'
	int len=strlen(m_szInitDir);
	if (m_szInitDir[len-1] != '\\')
		strcat(m_szInitDir,"\\");

	return true;
}

bool CBrowseDir::BeginBrowse(const char *filespec)
{
	ProcessDir(m_szInitDir,NULL);//空语句？？
	if(strchr(filespec,'\\')!=NULL)//文件夹下有文件
		return BrowseDir(filespec,"*.*");
	return BrowseDir(m_szInitDir,filespec);
}

bool CBrowseDir::BrowseDir(const char *dir,const char *filespec)
{
	_chdir(dir);//更改当前工作目录，更改成功返回0，失败返回-1。

	//首先查找dir中符合要求的文件
	long hFile;
	_finddata_t fileinfo;//定义结构体变量fileinfo。
	if ((hFile=_findfirst(filespec,&fileinfo)) != -1)//查找所有名为image（argv[1]）的文件
	{
		do
		{
			//检查是不是目录
			//如果不是,则进行处理
			if (!(fileinfo.attrib & _A_SUBDIR))
			{
				if (!ProcessFile(fileinfo.name))
					return false;
			}
			else{
				extern std::string export_path;
				std::string tmp(dir);
				if(dir[strlen(dir)-1]!='\\')
					tmp+='\\';
				tmp+=fileinfo.name;
				if(tmp.find(export_path)!=std::string::npos||
					export_path.find(tmp)!=std::string::npos)
					continue;
				if (strcmp(fileinfo.name,".") != 0 && strcmp
					(fileinfo.name,"..") != 0){
						std::string str(dir);
						if(dir[strlen(dir)-1]!='\\')
							str+='\\';
						str+=fileinfo.name;
						BrowseDir(str.c_str(),"*.*");
						_chdir(dir);
				}
			}
		} while (_findnext(hFile,&fileinfo) == 0);
		_findclose(hFile);
	}
#if 0
	//查找dir中的子目录
	//因为在处理dir中的文件时，派生类的ProcessFile有可能改变了
	//当前目录，因此还要重新设置当前目录为dir。
	//执行过_findfirst后，可能系统记录下了相关信息，因此改变目录
	//对_findnext没有影响。
	_chdir(dir);
	if ((hFile=_findfirst("*.*",&fileinfo)) != -1)
	{
		do
		{
			//检查是不是目录
			//如果是,再检查是不是 . 或 .. 
			//如果不是,进行迭代
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name,".") != 0 && strcmp
					(fileinfo.name,"..") != 0)
				{
					char subdir[_MAX_PATH];
					strcpy(subdir,dir);
					strcat(subdir,fileinfo.name);
					strcat(subdir,"\\");
					ProcessDir(subdir,dir);
					if (!BrowseDir(subdir,filespec))
						return false;
				}
			}
		} while (_findnext(hFile,&fileinfo) == 0);
		_findclose(hFile);

	}
#endif
	return true;
}

bool CBrowseDir::ProcessFile(const char *filename)
{
	return true;
}

void CBrowseDir::ProcessDir(const char 
	*currentdir,const char *parentdir)
{
}



#if 0
void main()
{
	//获取目录名
	char buf[256];
	printf("请输入要统计的目录名:");
	gets(buf);

	//构造类对象
	CStatDir statdir;

	//设置要遍历的目录
	if (!statdir.SetInitDir(buf))
	{
		puts("目录不存在。");
		return;
	}

	//开始遍历
	statdir.BeginBrowse("*.*");
	printf("文件总数: %d\n子目录总数:%d\n",statdir.GetFileCount(),statdir.GetSubdirCount());
} 

#endif



//把该代码复制到main函数

//从CBrowseDir派生出的子类，用来统计目录中的文件及子目录个数
class CStatDir:public CBrowseDir
{
protected:
	//覆写虚函数ProcessFile，每调用一次，文件个数加1
	virtual bool ProcessFile(const char *filename)
	{
		const char *p=strchr(filename,'.');
		if(p==NULL) return CBrowseDir::ProcessFile(filename);
#define FORMAT_COUNT 4
		static const char * formats[]={".jpg",".bmp",".png",".jpeg"};
		int i;
		for(i=0;i<FORMAT_COUNT;i++){
			if(strcmp(p,formats[i])==0)
				break;
		}
		if(i!=FORMAT_COUNT){
			//添加处理代码
			loop(filename);
		}
		return true;
	}
};