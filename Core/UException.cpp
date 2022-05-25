#ifndef UEXCEPTION_CPP
#define UEXCEPTION_CPP

#define __STDC_WANT_LIB_EXT1__ 1

#include <typeinfo>
#include <time.h>
#include <memory.h>
#include "UException.h"
#include "USupport.h"
#include "rdk_system.h"
//#include "ExceptionDispatcher.h"

namespace RDK {

// --------------------------
// ������������ � �����������
// --------------------------
UException::UException(void)
: Number(0), Type(0), ExLineNumber(-1)
{
 unsigned long long seconds(0);
 GetTimeOfDayInMicroseconds(seconds, TimeMsecs);
 Time=seconds;
}

UException::UException(const UException &copy)
: Number(copy.Number), Type(copy.Type), ExFileName(copy.ExFileName), ExLineNumber(copy.ExLineNumber),
  ObjectName(copy.ObjectName), InfoMessageString(copy.InfoMessageString)
{
// Number=copy.Number;
 Time=copy.Time;
 TimeMsecs=copy.TimeMsecs;
}


UException::~UException(void) throw()
{

}
// --------------------------


// --------------------------
// ������ ��������� ������� ����������
// --------------------------
// ���������� ����� ����������
int UException::GetNumber(void) const
{
 return Number;
}

// ��� ����������
int UException::GetType(void) const
{
 return Type;
}

// ����� ������������� (���������) ����������
std::time_t UException::GetTime(void) const
{
 return Time;
}

void UException::SetTime(std::time_t ex_time)
{
 Time=ex_time;
}

/// ��� ����� � ������� ��������� ����������
std::string UException::GetExFileName(void) const
{
 return ExFileName;
}

void UException::SetExFileName(const std::string &value)
{
 ExFileName=value;
}

/// ������ �� ������� ��������� ����������
int UException::GetExLineNumber(void) const
{
 return ExLineNumber;
}

void UException::SetExLineNumber(int value)
{
 ExLineNumber=value;
}

/// ��� ������� ���������������� ���������
std::string UException::GetObjectName(void) const
{
 return ObjectName;
}

void UException::SetObjectName(const std::string &value)
{
 ObjectName=value;
}

/// ��������� ���������� ����������� �� ���������� ������
const std::string& UException::GetMessage(void) const
{
 return Message;
}

void UException::SetMessage(const std::string& value)
{
 Message=value;
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
/// ���������� ������ ���� �� ����������
char const * UException::what() const throw()
{
 Message=GenerateLogPrefix();
 Message+=CreateLogMessage();
 Message+=InfoMessageString;
 return Message.c_str();
}

/// ����������� ���� ������������ ��� ������������, � ������������ ������� ��������� ������
void UException::Wrap(const UException &ex, const std::string &new_message)
{
 Number=ex.Number;
 Time=ex.Time;
 ExFileName=ex.ExFileName;
 ExLineNumber=ex.ExLineNumber;
 ObjectName=ex.ObjectName;
 InfoMessageString=ex.CreateLogMessage()+ex.InfoMessageString+new_message;
}

void UException::Wrap(const UException &ex, const std::string &new_message, int number)
{
 Wrap(ex, new_message);
 Number=number;
}

void UException::Wrap(const UException &ex, const std::string &new_message, int number, int new_type)
{
 Wrap(ex, new_message, number);
 Type=new_type;
}

std::string UException::CreateLogMessage(void) const
{
 std::string result;
 return result;
};

std::string UException::GenerateLogPrefix(void) const
{
 std::string result;

 std::time_t ex_time=GetTime();
 tm time_result;
 tm* time_struct(0);

#if defined(_MSC_VER)
 localtime_s(&time_result,&ex_time);
 time_struct=&time_result;
#elif defined(__STDC_LIB_EXT1__)
 time_struct=localtime_s(&ex_time,&time_result);
#else
 time_struct=localtime(&ex_time); // TODO: Possible unsafe!!
 memcpy(&time_result,time_struct,sizeof(time_result));
#endif

 if(!time_struct)
  return result;

 result+=sntoa(time_struct->tm_mday,2);
 result+="/";
 result+=sntoa(time_struct->tm_mon+1,2);
 result+="/";
 result+=sntoa(time_struct->tm_year+1900,4);
 result+=" ";

 result+=sntoa(time_struct->tm_hour,2);
 result+=":";
 result+=sntoa(time_struct->tm_min,2);
 result+=":";
 result+=sntoa(time_struct->tm_sec,2);
 result+=".";
 result+=sntoa(TimeMsecs/1000,3);
 result+=" ";

 result+=sntoa(GetType());
 result+="> [";
 if(Number != 0)
 {
  result+=sntoa(Number);
  result+=": ";
 }
 result+=typeid(*this).name();
 result+="]";

 if(!ExFileName.empty())
 {
  result+="> ";
  result+=ExFileName;
  result+=":";
  result+=sntoa(ExLineNumber);
  result+="> ";
 }

 if(!ObjectName.empty())
 {
  result+=" ";
  result+=ObjectName;
  result+="> ";
 }

 return result;
}
// --------------------------





/* ��������� ������ (��������� �� 0 ���������� � �.�.) */
//class EFatal: public UException

// --------------------------
// ������������ � �����������
// --------------------------
EFatal::EFatal(void)
{
 Type=RDK_EX_FATAL;
}

EFatal::EFatal(const EFatal &copy)
 : UException(copy)
{

}

EFatal::~EFatal(void) throw()
{

}
// --------------------------



/* ������, �������������� ������������� */
//class EError: public UException
// --------------------------
// ������������ � �����������
// --------------------------
EError::EError(void)
{
 Type=RDK_EX_ERROR;
}

EError::EError(const EError &copy)
 : UException(copy)
{

}

EError::~EError(void) throw()
{

}
// --------------------------



/* �������������� (�������� �� ������������� ������������� ��������) */
//class EWarning: public UException
// --------------------------
// ������������ � �����������
// --------------------------
EWarning::EWarning(void)
{
 Type=RDK_EX_WARNING;
}

EWarning::EWarning(const EWarning &copy)
 : UException(copy)
{

}

EWarning::~EWarning(void) throw()
{

}
// --------------------------


/* �������������� ���������, ������ ������� ������������ ������������� */
//class EInfo: public UException
// --------------------------
// ������������ � �����������
// --------------------------
EInfo::EInfo(void)
{
 Type=RDK_EX_INFO;
}

EInfo::EInfo(const EInfo &copy)
 : UException(copy)
{


}

EInfo::~EInfo(void) throw()
{

}
// --------------------------

/* ���������� ���������, ������ ������� ������������ ������������� */
// --------------------------
// ������������ � �����������
// --------------------------
EDebug::EDebug(void)
{
 Type=RDK_EX_DEBUG;
}

EDebug::EDebug(const EDebug &copy)
 : UException(copy)
{

}

EDebug::~EDebug(void) throw()
{

}
// --------------------------

// --------------------------
// ������������ � �����������
// --------------------------
EApp::EApp(void)
{
 Type=RDK_EX_APP;
}

EApp::EApp(const EApp &copy)
 : UException(copy)
{
}

EApp::~EApp(void) throw()
{
}
// --------------------------

/* ������ �������������� ������ � ����� */
// --------------------------
// ������������ � �����������
// --------------------------
EStrToNumber::EStrToNumber(const std::string &str)
 : Str(str)
{

}

EStrToNumber::~EStrToNumber(void) throw()
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string EStrToNumber::CreateLogMessage(void) const
{
 return EError::CreateLogMessage()+string(" Str=")+Str;
}
// --------------------------

// ����������, ��������� � ����������������
//struct EIdError
// --------------------------
// ������������ � �����������
// --------------------------
EIdError::EIdError(int id) : Id(id)
{
}

EIdError::~EIdError(void) throw()
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string EIdError::CreateLogMessage(void) const
{
 return EError::CreateLogMessage()+string(" Id=")+sntoa(Id);
}
// --------------------------

// ����������, ��������� � �������
//struct ENameError
// --------------------------
// ������������ � �����������
// --------------------------
ENameError::ENameError(const std::string &name) : Name(name)
{
}

ENameError::~ENameError(void) throw()
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string ENameError::CreateLogMessage(void) const
{
 return EError::CreateLogMessage()+string(" Name=")+Name;
}
// --------------------------

// ����������, ��������� � ���������
// --------------------------
// ������������ � �����������
// --------------------------
EIndexError::EIndexError(int index) : Index(index)
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string EIndexError::CreateLogMessage(void) const
{
 return EError::CreateLogMessage()+string(" Index=")+sntoa(Index);
}
// --------------------------

// --------------------------
// ������������ � �����������
// --------------------------
ESystemException::ESystemException(void)
{

}

ESystemException::ESystemException(const std::string &info)
: Info(info)
{

}

ESystemException::ESystemException(const ESystemException &copy)
: EFatal(copy), Info(copy.Info)
{

}

ESystemException::~ESystemException(void) throw()
{

}
// --------------------------


// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string ESystemException::CreateLogMessage(void) const
{
 return EFatal::CreateLogMessage()+Info;
}
// --------------------------

// --------------------------
// ������������ � �����������
// --------------------------
EFunctionReturnFalse::EFunctionReturnFalse(const std::string &file_name, const std::string &function_name, int line)
 : FileName(file_name), FunctionName(function_name), Line(line)
{

}

EFunctionReturnFalse::~EFunctionReturnFalse(void) throw()
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string EFunctionReturnFalse::CreateLogMessage(void) const
{
 return EError::CreateLogMessage()+std::string(" File=")+FileName+
			std::string(" Function=")+FunctionName+
			std::string(" Line=")+sntoa(Line);
}
// --------------------------


// --------------------------
// ������������ � �����������
// --------------------------
EFunctionReturnError::EFunctionReturnError(const std::string &file_name, const std::string &function_name, int line, int code)
 : FileName(file_name), FunctionName(function_name), Line(line), Code(code)
{

}

EFunctionReturnError::~EFunctionReturnError(void) throw()
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string EFunctionReturnError::CreateLogMessage(void) const
{
 return EError::CreateLogMessage()+std::string(" File=")+FileName+
			std::string(" Function=")+FunctionName+
			std::string(" Line=")+sntoa(Line)+
			std::string(" Code=")+sntoa(Code);
}
// --------------------------

// ���������� � ������� ������� ������ ��� ��������� ������
// --------------------------
// ������������ � �����������
// --------------------------
EStringFatal::EStringFatal(const std::string &str, int number)
 : Str(str)
{
 Number=number;
}

EStringFatal::~EStringFatal(void) throw()
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string EStringFatal::CreateLogMessage(void) const
{
 return EFatal::CreateLogMessage()+std::string(" ")+Str;
}


// ���������� � ������� ������� ������ ��� ��������� ������
// --------------------------
// ������������ � �����������
// --------------------------
EStringError::EStringError(const std::string &str, int number)
 : Str(str)
{
 Number=number;
}

EStringError::~EStringError(void) throw()
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string EStringError::CreateLogMessage(void) const
{
 return EError::CreateLogMessage()+std::string(" ")+Str;
}

// --------------------------
// ���������� � ������� ������� ������ ��� ��������������
// --------------------------
// ������������ � �����������
// --------------------------
EStringWarning::EStringWarning(const std::string &str, int number)
 : Str(str)
{
 Number=number;
}

EStringWarning::~EStringWarning(void) throw()
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string EStringWarning::CreateLogMessage(void) const
{
 return EWarning::CreateLogMessage()+std::string(" ")+Str;
}
// --------------------------


// ���������� � ������� ������� ������ ��� �������������� ���������
// --------------------------
// ������������ � �����������
// --------------------------
EStringInfo::EStringInfo(const std::string &str, int number)
 : Str(str)
{
 Number=number;
}

EStringInfo::~EStringInfo(void) throw()
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string EStringInfo::CreateLogMessage(void) const
{
 return EInfo::CreateLogMessage()+std::string(" ")+Str;
}
// --------------------------

// ���������� � ������� ������� ������ ��� ���������� ���������
// --------------------------
// ������������ � �����������
// --------------------------
EStringDebug::EStringDebug(const std::string &str, int number)
 : Str(str)
{
 Number=number;
}

EStringDebug::~EStringDebug(void) throw()
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string EStringDebug::CreateLogMessage(void) const
{
 return EDebug::CreateLogMessage()+std::string(" ")+Str;
}
// --------------------------

/// ���������� ��� �������������� � ��������������� ������� ����������
// --------------------------
// ������������ � �����������
// --------------------------
EStringApp::EStringApp(const std::string &str, int number)
 : Str(str)
{
 Number=number;
}

EStringApp::~EStringApp(void) throw()
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string EStringApp::CreateLogMessage(void) const
{
 return EApp::CreateLogMessage()+std::string(" ")+Str;
}
// --------------------------

// --------------------------
// ������������ � �����������
// --------------------------
UExceptionUnhandled::UExceptionUnhandled(const std::string &file_name, int line, const std::string &function_name)
 : FileName(file_name), FunctionName(function_name), Line(line)
{

}

UExceptionUnhandled::~UExceptionUnhandled(void) throw()
{
}
// --------------------------

// --------------------------
// ������ ������������ ����
// --------------------------
// ��������� ������ ���� �� ����������
std::string UExceptionUnhandled::CreateLogMessage(void) const
{
 return EFatal::CreateLogMessage()+std::string(" File=")+FileName+
			std::string(":")+sntoa(Line)+
			std::string(" Func=")+FunctionName;
}
// --------------------------


}
#endif

