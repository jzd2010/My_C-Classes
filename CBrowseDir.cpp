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
	//�õ�ǰĿ¼��ʼ��m_szInitDir
	getcwd(m_szInitDir,_MAX_PATH);

	//���Ŀ¼�����һ����ĸ����'\',����������һ��'\'
	int len=strlen(m_szInitDir);
	if (m_szInitDir[len-1] != '\\')
		strcat(m_szInitDir,"\\");
}

bool CBrowseDir::SetInitDir(const char *dir)
{
	//�Ȱ�dirת��Ϊ����·��
	if (_fullpath(m_szInitDir,dir,_MAX_PATH) == NULL)
		return false;

	//�ж�Ŀ¼�Ƿ����
	if (_chdir(m_szInitDir) != 0)
		return false;

	//���Ŀ¼�����һ����ĸ����'\',����������һ��'\'
	int len=strlen(m_szInitDir);
	if (m_szInitDir[len-1] != '\\')
		strcat(m_szInitDir,"\\");

	return true;
}

bool CBrowseDir::BeginBrowse(const char *filespec)
{
	ProcessDir(m_szInitDir,NULL);//����䣿��
	if(strchr(filespec,'\\')!=NULL)//�ļ��������ļ�
		return BrowseDir(filespec,"*.*");
	return BrowseDir(m_szInitDir,filespec);
}

bool CBrowseDir::BrowseDir(const char *dir,const char *filespec)
{
	_chdir(dir);//���ĵ�ǰ����Ŀ¼�����ĳɹ�����0��ʧ�ܷ���-1��

	//���Ȳ���dir�з���Ҫ����ļ�
	long hFile;
	_finddata_t fileinfo;//����ṹ�����fileinfo��
	if ((hFile=_findfirst(filespec,&fileinfo)) != -1)//����������Ϊimage��argv[1]�����ļ�
	{
		do
		{
			//����ǲ���Ŀ¼
			//�������,����д���
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
	//����dir�е���Ŀ¼
	//��Ϊ�ڴ���dir�е��ļ�ʱ���������ProcessFile�п��ܸı���
	//��ǰĿ¼����˻�Ҫ�������õ�ǰĿ¼Ϊdir��
	//ִ�й�_findfirst�󣬿���ϵͳ��¼���������Ϣ����˸ı�Ŀ¼
	//��_findnextû��Ӱ�졣
	_chdir(dir);
	if ((hFile=_findfirst("*.*",&fileinfo)) != -1)
	{
		do
		{
			//����ǲ���Ŀ¼
			//�����,�ټ���ǲ��� . �� .. 
			//�������,���е���
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
	//��ȡĿ¼��
	char buf[256];
	printf("������Ҫͳ�Ƶ�Ŀ¼��:");
	gets(buf);

	//���������
	CStatDir statdir;

	//����Ҫ������Ŀ¼
	if (!statdir.SetInitDir(buf))
	{
		puts("Ŀ¼�����ڡ�");
		return;
	}

	//��ʼ����
	statdir.BeginBrowse("*.*");
	printf("�ļ�����: %d\n��Ŀ¼����:%d\n",statdir.GetFileCount(),statdir.GetSubdirCount());
} 

#endif



//�Ѹô��븴�Ƶ�main����

//��CBrowseDir�����������࣬����ͳ��Ŀ¼�е��ļ�����Ŀ¼����
class CStatDir:public CBrowseDir
{
protected:
	//��д�麯��ProcessFile��ÿ����һ�Σ��ļ�������1
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
			//��Ӵ������
			loop(filename);
		}
		return true;
	}
};