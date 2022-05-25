#ifndef UEXCEPTION_H
#define UEXCEPTION_H

#include <ctime>
#include <string>
#include "initdll_defs.h"

/// Exception Type
#define RDK_EX_UNKNOWN 0 // Unknown exception
#define RDK_EX_FATAL 1 // Fatal error (correction impossible)
#define RDK_EX_ERROR 2 // Correctable error
#define RDK_EX_WARNING 3 // Warning (performance, possible errors etc)
#define RDK_EX_INFO 4 // Information (port open, client connected etc)
#define RDK_EX_APP 5 // Application-defined event (high-level errors, etc)
#define RDK_EX_DEBUG 6 // Debug messages (can be switched off)

#define RDK_SYS_DEBUG_NONE 0
#define RDK_SYS_DEBUG_CALC 1
#define RDK_SYS_DEBUG_RESET 2
#define RDK_SYS_DEBUG_PROPERTIES 4
#define RDK_SYS_DEBUG_PARAMETERS 8
#define RDK_SYS_DEBUG_STATES 16
#define RDK_SYS_DEBUG_INPUTS 32
#define RDK_SYS_DEBUG_OUTPUTS 64

#define RDK_SYS_MESSAGE_ENTER 1
#define RDK_SYS_MESSAGE_EXIT_OK 2
#define RDK_SYS_MESSAGE_EXIT_ININIT_FAIL 4
#define RDK_SYS_MESSAGE_NEW_CALC_ITERATION 8


namespace RDK {

/* ������� ����� ���������� */
#if defined(_MSC_VER)
    #pragma warning( push )
    #pragma warning( disable : 4275)
#endif
class RDK_LIB_TYPE UException: public std::exception
{
protected: // ������ ����������
// ����� ����������
int Number;

/// ��� ����������
int Type;

// ����� ������������� ����������
std::time_t Time;
unsigned long long TimeMsecs;

/// ��� ����� � ������� ��������� ����������
mutable std::string ExFileName;

/// ������ �� ������� ��������� ����������
mutable int ExLineNumber;

/// ��� ������� ���������������� ���������
mutable std::string ObjectName;

/// ��������� ����������
mutable std::string Message;

/// ������, ���������� ������������ ���������� � ������������
mutable std::string InfoMessageString;

public: // ������
// --------------------------
// ������������ � �����������
// --------------------------
UException(void);
UException(const UException &copy);
virtual ~UException(void) throw();
// --------------------------

// --------------------------
// ������ ���������� ������ �������
// --------------------------
// --------------------------

// --------------------------
// ������ ��������� ������� ����������
// --------------------------
// ���������� ����� ����������
int GetNumber(void) const;

// ��� ����������
int GetType(void) const;

// ����� ������������� (���������) ����������
std::time_t GetTime(void) const;
void SetTime(std::time_t ex_time);

/// ��� ����� � ������� ��������� ����������
std::string GetExFileName(void) const;
void SetExFileName(const std::string &value);

/// ������ �� ������� ��������� ����������
int GetExLineNumber(void) const;
void SetExLineNumber(int value);

/// ��� ������� ���������������� ���������
std::string GetObjectName(void) const;
void SetObjectName(const std::string &value);

/// ��������� ���������� ����������� �� ���������� ������
const std::string& GetMessage(void) const;
void SetMessage(const std::string& value);
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
/// ���������� ������ ���� �� ����������
virtual char const * what() const throw();

/// ����������� ���� ������������ ��� ������������, � ������������ ������� ��������� ������
void Wrap(const UException &ex, const std::string &new_message);
void Wrap(const UException &ex, const std::string &new_message, int number);
void Wrap(const UException &ex, const std::string &new_message, int number, int new_type);

protected:
/// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
virtual std::string GenerateLogPrefix(void) const;
// --------------------------
};
#if defined(_MSC_VER)
    #pragma warning( pop )
#endif

/* ��������� ������ (��������� �� 0 ���������� � �.�.) */
struct RDK_LIB_TYPE EFatal: public UException
{
// --------------------------
// ������������ � �����������
// --------------------------
EFatal(void);
EFatal(const EFatal &copy);
virtual ~EFatal(void) throw();
// --------------------------

};

/* ������, �������������� ������������� */
struct RDK_LIB_TYPE EError: public UException
{
// --------------------------
// ������������ � �����������
// --------------------------
EError(void);
EError(const EError &copy);
virtual ~EError(void) throw();
// --------------------------

};

/* �������������� (�������� �� ������������� ������������� ��������) */
struct RDK_LIB_TYPE EWarning: public UException
{
// --------------------------
// ������������ � �����������
// --------------------------
EWarning(void);
EWarning(const EWarning &copy);
virtual ~EWarning(void) throw();
// --------------------------

};

/* �������������� ���������, ������ ������� ������������ ������������� */
struct RDK_LIB_TYPE EInfo: public UException
{
// --------------------------
// ������������ � �����������
// --------------------------
EInfo(void);
EInfo(const EInfo &copy);
virtual ~EInfo(void) throw();
// --------------------------
};

/* ���������� ���������, ������ ������� ������������ ������������� */
struct RDK_LIB_TYPE EDebug: public UException
{
// --------------------------
// ������������ � �����������
// --------------------------
EDebug(void);
EDebug(const EDebug &copy);
virtual ~EDebug(void) throw();
// --------------------------
};

/// ���������� ��� �������������� � ��������������� ������� ����������
struct RDK_LIB_TYPE EApp: public UException
{
// --------------------------
// ������������ � �����������
// --------------------------
EApp(void);
EApp(const EApp &copy);
virtual ~EApp(void) throw();
// --------------------------
};

/* ������ �������������� ������ � ����� */
struct RDK_LIB_TYPE EStrToNumber: public EError
{
std::string Str; // ������

// --------------------------
// ������������ � �����������
// --------------------------
explicit EStrToNumber(const std::string &str);
virtual ~EStrToNumber(void) throw();
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};

// ����������, ��������� � ����������������
struct RDK_LIB_TYPE EIdError: public EError
{
// �������������, ��������� ����������
int Id;

// --------------------------
// ������������ � �����������
// --------------------------
explicit EIdError(int id);
virtual ~EIdError(void) throw();
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};

// ����������, ��������� � �������
struct RDK_LIB_TYPE ENameError: public EError
{
// �������������, ��������� ����������
std::string Name;

// --------------------------
// ������������ � �����������
// --------------------------
explicit ENameError(const std::string &name);
virtual ~ENameError(void) throw();
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};


// Id �� ������
struct RDK_LIB_TYPE EIdNotExist: public EIdError
{
explicit EIdNotExist(int id) : EIdError(id) {};
};

// ��� �� �������
struct RDK_LIB_TYPE ENameNotExist: public ENameError
{
explicit ENameNotExist(const std::string &name) : ENameError(name) {};
};

// Id ��� ����������
struct RDK_LIB_TYPE EIdAlreadyExist: public EIdError
{
explicit EIdAlreadyExist(int id) : EIdError(id) {};
};

// ��� ��� ����������
struct RDK_LIB_TYPE ENameAlreadyExist: public ENameError
{
explicit ENameAlreadyExist(const std::string &name) : ENameError(name) {};
};

// Id �� ��������� (forbidden id)
struct RDK_LIB_TYPE EForbiddenId: public EIdError
{
explicit EForbiddenId(int id) : EIdError(id) {};
};

// Id �� ���������
struct RDK_LIB_TYPE EInvalidId: public EIdError
{
explicit EInvalidId(int id) : EIdError(id) {};
};

// ����������, ��������� � ���������
struct RDK_LIB_TYPE EIndexError: public EError
{
// ������, ��������� ����������
int Index;

// --------------------------
// ������������ � �����������
// --------------------------
explicit EIndexError(int index);
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};

// Id �� ���������
struct RDK_LIB_TYPE EInvalidIndex: public EIndexError
{
explicit EInvalidIndex(int index) : EIndexError(index) {};
};


struct RDK_LIB_TYPE ESystemException: public EFatal
{
// �������������� ���������� �� ����������
std::string Info;

// --------------------------
// ������������ � �����������
// --------------------------
ESystemException(void);
explicit ESystemException(const std::string &info);
ESystemException(const ESystemException &copy);
virtual ~ESystemException(void) throw();
// --------------------------


// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------

};

// ����������, ��������� � ���������� ������������ �������� �������
struct RDK_LIB_TYPE EFunctionReturnFalse: public EError
{
// ��� �����
std::string FileName;

// ��� ������� � ������� ��������� ������
std::string FunctionName;

// ������ � ���������
int Line;

// --------------------------
// ������������ � �����������
// --------------------------
EFunctionReturnFalse(const std::string &file_name, const std::string &function_name, int line);
virtual ~EFunctionReturnFalse(void) throw();
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};

// ����������, ��������� � ���������� ������������ �������� �������
struct RDK_LIB_TYPE EFunctionReturnError: public EError
{
// ��� �����
std::string FileName;

// ��� ������� � ������� ��������� ������
std::string FunctionName;

// ������ � ���������
int Line;

// ��� ������
int Code;

// --------------------------
// ������������ � �����������
// --------------------------
EFunctionReturnError(const std::string &file_name, const std::string &function_name, int line, int code);
virtual ~EFunctionReturnError(void) throw();
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};

// ���������� � ������� ������� ������ ��� ��������� ������
struct RDK_LIB_TYPE EStringFatal: public EFatal
{
std::string Str;

// --------------------------
// ������������ � �����������
// --------------------------
EStringFatal(const std::string &str, int number=0);
virtual ~EStringFatal(void) throw();
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};

// ���������� � ������� ������� ������ ��� ������
struct RDK_LIB_TYPE EStringError: public EError
{
std::string Str;

// --------------------------
// ������������ � �����������
// --------------------------
EStringError(const std::string &str, int number=0);
virtual ~EStringError(void) throw();
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};

// ���������� � ������� ������� ������ ��� ��������������
struct RDK_LIB_TYPE EStringWarning: public EWarning
{
std::string Str;

// --------------------------
// ������������ � �����������
// --------------------------
EStringWarning(const std::string &str, int number=0);
virtual ~EStringWarning(void) throw();
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};

// ���������� � ������� ������� ������ ��� �������������� ���������
struct RDK_LIB_TYPE EStringInfo: public EInfo
{
std::string Str;

// --------------------------
// ������������ � �����������
// --------------------------
EStringInfo(const std::string &str, int number=0);
virtual ~EStringInfo(void) throw();
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};

// ���������� � ������� ������� ������ ��� ���������� ���������
struct RDK_LIB_TYPE EStringDebug: public EDebug
{
std::string Str;

// --------------------------
// ������������ � �����������
// --------------------------
EStringDebug(const std::string &str, int number=0);
virtual ~EStringDebug(void) throw();
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};

/// ���������� ��� �������������� � ��������������� ������� ����������
struct RDK_LIB_TYPE EStringApp: public EApp
{
std::string Str;

// --------------------------
// ������������ � �����������
// --------------------------
EStringApp(const std::string &str, int number=0);
virtual ~EStringApp(void) throw();
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};



// ����������, ������� ���������� ����������
struct RDK_LIB_TYPE UExceptionUnhandled: public EFatal
{
// ��� �����
std::string FileName;

// ��� ������� � ������� ��������� ������
std::string FunctionName;

// ������ � ���������
int Line;

// --------------------------
// ������������ � �����������
// --------------------------
UExceptionUnhandled(const std::string &file_name, int line, const std::string &function_name);
virtual ~UExceptionUnhandled(void) throw();
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
virtual std::string CreateLogMessage(void) const;
// --------------------------
};


}

#endif

