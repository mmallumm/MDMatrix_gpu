#pragma once
#ifndef RDK_SYSTEM_WIN_CPP
#define RDK_SYSTEM_WIN_CPP

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <OleAuto.h>
#include "rdk_system.h"
//#include "USharedMemoryLoader.win.cpp"
//#include "UGenericMutex.win.cpp"
#include "UDllLoader.win.cpp"

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  116444736000000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  116444736000000000ULL
#endif


namespace RDK {


// ���������� ������� ����� � ������������� �� ���������� �������������� �������
// (������� �� ����������)
unsigned long long GetCurrentStartupTime(void)
{
 return GetTickCount();
}

// ���������� � seconds � useconds ������� �������� ������ � �����������,
// ��������� � ���������� �������������� �������
void GetTimeOfDayInMicroseconds(unsigned long long &seconds, unsigned long long &useconds)
{
 FILETIME ft;
 long long tmpres = 0;

 GetSystemTimeAsFileTime(&ft);

 tmpres |= ft.dwHighDateTime;
 tmpres <<= 32;
 tmpres |= ft.dwLowDateTime;

 // converting file time to unix epoch
 tmpres -= DELTA_EPOCH_IN_MICROSECS;
 tmpres /= 10;  /*convert into microseconds*/
 time_t secs=(int)(tmpres / 1000000UL);
 seconds = (unsigned long long)(tmpres / 1000000UL);
 useconds = (unsigned long long)(tmpres % 1000000UL);
}

// ��������� ������� �� ������� � �������������
unsigned long long CalcDiffTime(unsigned long long time1, unsigned long long time2)
{
 if(time1>time2)
  return time1-time2;
 else
  return time2-time1;
}

/// ���������� ��������� ����� � ���� (� ��������� �� �����������) �� ������ ������
double GetVariantLocalTime(void)
{
 SYSTEMTIME loc_time;
 GetLocalTime(&loc_time);
 double res=0;
 SystemTimeToVariantTime(&loc_time,&res);
 return res +  ((ONETHOUSANDMILLISECONDS/1000.0) * loc_time.wMilliseconds);
}


// �������� ������� �� �������� ����� �����������
void Sleep(int value)
{
 ::Sleep(value);
}                                  

// ������� �������
// ���������� 0 � ������ ������ ��� ���� ������� ��� ����������
// 1 - ���� ��� ���������� ���� � ����� ������
// 2 - ���� ����� ���� �� ����������
// 3 - ���� ��������� ������ ������
int CreateNewDirectory(const char* path)
{
 DWORD dwFileAttributes = GetFileAttributesA(path);
 if(dwFileAttributes == INVALID_FILE_ATTRIBUTES || (dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
 {
  if(!::CreateDirectoryA(path, 0))
  {
   if(GetLastError() == ERROR_PATH_NOT_FOUND)
    return 2;

   return 0;
  }
 }
 else
  return 1;

 return 0;
}         

//---------------------------------------------------------------------------


// �������� ������ ������ ��� ��������� �� ��������� ����
int FindFilesList(const std::string &path, const std::string &mask, bool isfile, std::vector<std::string> &results)
{
   results.clear();
   HANDLE findhandle;
 std::string filemask=path+mask;
 WIN32_FIND_DATAA finddata;
 std::string samplefilename;

 findhandle=FindFirstFileA(
    filemask.c_str(),    // pointer to name of file to search for
    &finddata     // pointer to returned information
    );

 if(findhandle != INVALID_HANDLE_VALUE)
 {
  do
  {      
    samplefilename=finddata.cFileName;
 //  samplefilename=AnsiString(ExtractFileName(finddata.cFileName)).c_str();
   std::string::size_type i=samplefilename.find_last_of("\\/");
   if(i != std::string::npos)
    samplefilename.erase(0,i+1);

   if(samplefilename != "." && samplefilename != ".." && ((!isfile && (finddata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || isfile)))
   {
    results.push_back(samplefilename);
   }

  } while(FindNextFileA(findhandle,&finddata));
 }
 FindClose(findhandle);
 return 0;
}

int RdkCopyFile(const std::string &source_file, const std::string &dest_file)
{
 DWORD error=0;

 if(!CopyFileExA(source_file.c_str(), dest_file.c_str(),0,0, FALSE, COPY_FILE_ALLOW_DECRYPTED_DESTINATION | COPY_FILE_OPEN_SOURCE_FOR_WRITE))
 {
  error=GetLastError();
  return error;
 }

 return 0;
}

/// ���������� ����
int RdkMoveFile(const std::string &source_file, const std::string &dest_file)
{
 DWORD error=0;
 if(!MoveFileExA(source_file.c_str(), dest_file.c_str(),MOVEFILE_WRITE_THROUGH))
 {
  error=GetLastError();
  return error;
 }

 return 0;
}

int CopyDir(const std::string &source_dir, const std::string &dest_dir, const std::string &mask)
{
 std::vector<std::string> results;

 int res=FindFilesList(source_dir, mask, true, results);
 DWORD error=0;
 if(!res)
 {
  for(size_t i=0;i<results.size();i++)
  {
   DWORD err=RdkCopyFile(source_dir+results[i],dest_dir+results[i]);
   if(err != 0)
	error=err;
  }
 }
 return error;
}

/// ������� ������������ ����� � ���������� ���, ���� ������ � �������
void RdkDebuggerMessage(const std::string &message)
{
}

/// ������� ������� ���������� ������������ ��������� � �������� ��� ���� Load(dll_name)
RDK_LIB_TYPE UDllLoader* UCreateAndLoadDllLoader(const std::string &dll_name)
{
    UDllLoader * loader = new UDllLoaderWin(dll_name);
    loader->Load();
	return loader;
}

/// ������� ���������� ������� ���������� ������������ ����������, �� ��������� ����������
RDK_LIB_TYPE void UDestroyDllLoader(UDllLoader *handle)
{
    if (handle)
      delete handle;
}


unsigned long long GetMemoryUsedInfo(void)
{
#ifdef __BORLANDC__
 unsigned long long	result(0);
 PROCESS_MEMORY_COUNTERS mc;
 int cb=sizeof(PROCESS_MEMORY_COUNTERS);

 if(GetProcessMemoryInfo(GetCurrentProcess(), &mc, cb))
  result= mc.WorkingSetSize;
 return result;
#else
 return 0;
#endif
}

#ifdef __BORLANDC__
unsigned long long GetLargestFreeMemRegion(void* &AAddressOfLargest)
{
 TSystemInfo Si;
 LongWord P, dwRet;
 TMemoryBasicInformation Mbi;

 unsigned long long Result = 0;
 AAddressOfLargest = 0;
 GetSystemInfo(&Si);
 P = 0;
 while (P < LongWord(Si.lpMaximumApplicationAddress))
 {
  dwRet = VirtualQuery((void*)(P), &Mbi, sizeof(Mbi));
  if( (dwRet > 0) && (Mbi.State && MEM_FREE != 0))
  {
	  if (Result < Mbi.RegionSize)
	  {
		Result = Mbi.RegionSize;
		AAddressOfLargest = Mbi.BaseAddress;
	  }
	  P+=Mbi.RegionSize;
  }
  else
   P+=Si.dwPageSize;
 }
 return Result;
}
#else
unsigned long long GetLargestFreeMemRegion(LPVOID *lpBaseAddr)
{
    SYSTEM_INFO systemInfo;
    GetSystemInfo(&systemInfo);
    LPVOID p = 0;
    MEMORY_BASIC_INFORMATION mbi;
    SIZE_T largestSize = 0;
    while(p < systemInfo.lpMaximumApplicationAddress)
    {
        SIZE_T dwRet = VirtualQuery(p, &mbi, sizeof(mbi));
        if (dwRet > 0)
        {
            if (mbi.State && MEM_FREE)
            {
                if (largestSize < mbi.RegionSize)
                {
                    largestSize = mbi.RegionSize;
                    if (lpBaseAddr  != NULL)
                        *lpBaseAddr  = mbi.BaseAddress;
                }
            }
            p = (void*)((char*)(p) + mbi.RegionSize);
        }
        else
        {
			p = (void*)((char*)(p)+systemInfo.dwPageSize);
        }
    }
    return static_cast<unsigned long long>(largestSize);
}
#endif

/// ���������� ����� ������������ ����������� ������
/// ���� �� ������� ���������� �� ���������� false
bool ReadUsedMemoryInfo(unsigned long long &total_used_memory, unsigned long long &largest_free_block)
{
 total_used_memory=GetMemoryUsedInfo();
#ifdef __BORLANDC__
 void* AAddressOfLargest(0);
#else
 LPVOID* AAddressOfLargest(0);
#endif
 largest_free_block=GetLargestFreeMemRegion(AAddressOfLargest);
 return true;

}


}
#endif

