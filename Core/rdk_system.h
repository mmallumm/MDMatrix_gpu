#ifndef RDK_SYSTEM_H
#define RDK_SYSTEM_H   

#include <vector>
#include <string>
#include <locale>
#include "USharedMemoryLoader.h"
#include "UGenericMutex.h"
#include "UDllLoader.h"

#define ONETHOUSANDMILLISECONDS  .0000115740740740

namespace RDK {

extern "C++" {

/// ���������� ������� ����� � ������������� �� ���������� �������������� �������
/// (������� �� ����������)
RDK_LIB_TYPE unsigned long long GetCurrentStartupTime(void);

// ���������� � seconds � useconds ������� �������� ������ � �����������,
// ��������� � ���������� �������������� �������
RDK_LIB_TYPE void GetTimeOfDayInMicroseconds(unsigned long long &seconds, unsigned long long &useconds);

/// ��������� ������� �� ������� � �������������
RDK_LIB_TYPE unsigned long long CalcDiffTime(unsigned long long time1, unsigned long long time2);

/// ���������� ��������� ����� � ���� (� ��������� �� �����������) �� ������ ������
RDK_LIB_TYPE double GetVariantLocalTime(void);

/// �������� ������� �� �������� ����� �����������
RDK_LIB_TYPE void Sleep(int value);

/// ������� �������
/// ���������� 0 � ������ ������ ��� ���� ������� ��� ����������
/// 1 - ���� ��� ���������� ���� � ����� ������
/// 2 - ���� ����� ���� �� ����������
/// 3 - ���� ��������� ������ ������
RDK_LIB_TYPE int CreateNewDirectory(const char* path);

/// ������� ��� ����� � ����������
/// � ���� ����������, ���� � ��� ���� ������ �����
RDK_LIB_TYPE int DeleteDirectory(const char* path);

/// ������� ����
RDK_LIB_TYPE int RemoveFile(const char* path);

/// �������� ������ ������ ��� ��������� �� ��������� ����
RDK_LIB_TYPE int FindFilesList(const std::string &path, const std::string &mask, bool isfile, std::vector<std::string> &results);

/// �������� ����
RDK_LIB_TYPE int RdkCopyFile(const std::string &source_file, const std::string &dest_file);

/// ���������� ����
RDK_LIB_TYPE int RdkMoveFile(const std::string &source_file, const std::string &dest_file);

/// �������� ������� � ����������
RDK_LIB_TYPE int CopyDir(const std::string &source_dir, const std::string &dest_dir, const std::string &mask);

/// ������� ������������ ����� � ���������� ���, ���� ������ � �������
RDK_LIB_TYPE void RdkDebuggerMessage(const std::string &message);

/// ������� ������� ���������� ������������ ��������� � �������� ��� ���� Load(dll_name)
RDK_LIB_TYPE UDllLoader* UCreateAndLoadDllLoader(const std::string &dll_name);

/// ������� ���������� ������� ���������� ������������ ����������, �� ��������� ����������
RDK_LIB_TYPE void UDestroyDllLoader(UDllLoader *handle);

/// ���������� ����� ������������ ����������� ������
/// ���� �� ������� ���������� �� ���������� false
RDK_LIB_TYPE bool ReadUsedMemoryInfo(unsigned long long &total_used_memory, unsigned long long &largest_free_block);

/// ����������� ������� ������ � �������
/// 0 - �������� ��������� �� ���������
RDK_LIB_TYPE std::string& narrow(const std::wstring& wstr, std::string &result, unsigned codepage=0);

/// Deprecated
RDK_LIB_TYPE std::string& narrow(const std::wstring& wstr, const std::locale& loc, std::string &result);

/// ����������� ������� ������ � �������
/// 0 - �������� ��������� �� ���������
RDK_LIB_TYPE std::wstring& widen(const std::string& str, std::wstring &result, unsigned codepage=0);

/// Deprecated
RDK_LIB_TYPE std::wstring& widen(const std::string& str, const std::locale& loc, std::wstring &result);

}

}

#endif

