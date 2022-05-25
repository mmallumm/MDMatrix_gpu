#ifndef USHAREDMEMORYLOADER_H
#define USHAREDMEMORYLOADER_H

#include "initdll_defs.h"

#ifdef RDK_CALL
#define CALLING_CONVERSION RDK_CALL
#else
    #ifdef __BORLANDC__
        #define RDK_CALL __cdecl
    #else
        #ifdef __GNUC__
            #define RDK_CALL
        #else
            #define RDK_CALL __cdecl
        #endif
    #endif

#define CALLING_CONVERSION RDK_CALL
#endif

/// ���������� ����� ��������� ������� ������
typedef int (CALLING_CONVERSION *PUsm_GetNumPipes)(void);

/// ������ ����� ������� ������ �������
/// ��� ������� �� ������ ���������� �� ����� ������!
typedef void (CALLING_CONVERSION *PUsm_SetNumPipes)(int value);

/// ������ ������ ������ ������ 'pipe_index'
/// ����� �������� ������ ����������� 'mode', master (1) ��� slave (0)
/// (��������������� �� ������������� ����� ����� �� ������������)
/// �������� 'pipe_uid' ���������� ���������� ��������� ������������� ����� ������� ������ ��� ����� ������.
/// (��������������� �� ������������ �������������� ������� ������������)
/// ��� ������� �� ������ ���������� �� ����� ������ �� ���������� ������!
/// � ������ ����������� � ������ slave ������� �������� ���������� 'pipe_byte_size', � � ��� ������������
/// ������ ��������� ������ ��� ����� ������
/// ���������� 0 � ������ ������, ����� ������������� �����
typedef int (CALLING_CONVERSION *PUsm_InitPipe)(int pipe_index, int &pipe_byte_size, int mode, const char *pipe_uid);

/// ����������� �� ������
typedef int (CALLING_CONVERSION *PUsm_UnInitPipe)(int pipe_index);

/// ���������, ��������������� �� �����
/// Sic! ���������� 0 � ������, ���� ������ ���������������
/// ��� ��������� �� ������ < 0.
typedef int (CALLING_CONVERSION *PUsm_IsPipeInit)(int pipe_index);

/// ���������� ������ ����� ������ ������
typedef int (CALLING_CONVERSION *PUsm_GetPipeSize)(int pipe_index);

/// ���������� ������ � ����� ������, ������� �� �� ������ 'buffer'
/// ���������� ����� ���������� ���� ��� ������������� ����� � ������ ������
typedef int (CALLING_CONVERSION *PUsm_WriteData)(int pipe_index, const char* buffer, int buffer_byte_size);

/// ������ ������ �� ����� ������, ������� �� � ����� 'buffer', �������� buffer_byte_size
/// ������ ������ ���� ��������
/// ���������� ����� ����������� ���� ��� ������������� ����� � ������ ������
typedef int (CALLING_CONVERSION *PUsm_ReadData)(int pipe_index, char* buffer, int buffer_byte_size);

extern RDK_LIB_TYPE PUsm_GetNumPipes Usm_GetNumPipes;
extern RDK_LIB_TYPE PUsm_SetNumPipes Usm_SetNumPipes;
extern RDK_LIB_TYPE PUsm_InitPipe Usm_InitPipe;
extern RDK_LIB_TYPE PUsm_UnInitPipe Usm_UnInitPipe;
extern RDK_LIB_TYPE PUsm_IsPipeInit Usm_IsPipeInit;
extern RDK_LIB_TYPE PUsm_GetPipeSize Usm_GetPipeSize;
extern RDK_LIB_TYPE PUsm_WriteData Usm_WriteData;
extern RDK_LIB_TYPE PUsm_ReadData Usm_ReadData;

/// ��������� ����������
RDK_LIB_TYPE int LoadUSharedMemoryLibrary(const char *library_file_name);

/// ��������� ����������
RDK_LIB_TYPE int UnLoadUSharedMemoryLibrary(void);

#endif // USHAREDMEMORYLOADER_H
