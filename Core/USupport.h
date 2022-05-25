/* ***********************************************************
@Copyright Alexsandr V. Bakhshiev, 2005.
E-mail:        alexab@ailab.ru
Url:           http://ailab.ru

This file is part of the project: RDK

File License:       New BSD License
Project License:    New BSD License
See file license.txt for more information
*********************************************************** */
#ifndef USUPPORT_H
#define USUPPORT_H

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <locale>
#include <limits>
#include "UPtr.h"

#ifndef u_min
#define u_min(a,b) (a<b?a:b)
#endif

#ifndef u_max
#define u_max(a,b) (a>b?a:b)
#endif

#ifndef M_E
 #define M_E 2.71828182845904523536
#endif

#ifndef M_LOG2E
#define M_LOG2E 1.44269504088896340736
#endif

#ifndef M_LOG10E
#define M_LOG10E 0.434294481903251827651
#endif

#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

#ifndef M_PI_4
#define M_PI_4 0.785398163397448309616
#endif

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538
#endif

#ifndef M_2_PI
#define M_2_PI 0.636619772367581343076
#endif

#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524401
#endif

#ifndef ISNAN
#define ISNAN(x) ((x) != (x))
#endif


namespace RDK {

using namespace std;

// C�������� ������
//extern string ustrbuf;
//extern wstring uwstrbuf;

/*
 ������� ���������� ��������� �� ���������� �����
 �������� ������ ���������� ��� ��������� ������ ������� ����������
*/

/// �������� �� NaN
template<typename NumT>
bool is_nan(NumT n)
{
 return n != n;
}

/// �������� �� Inf
template<typename NumT>
bool is_inf(NumT n)
{
 return ((n==std::numeric_limits<NumT>::infinity()) || (n==-std::numeric_limits<NumT>::infinity()));
}


// �������, ������������� ����� � ������
template<typename CharT, typename NumT>
basic_string<CharT>& ntoa(NumT n, basic_string<CharT> &buf)
{
 basic_stringstream<CharT> stream;
 stream<<n;
 return buf=stream.str();
}

template<typename NumT>
string sntoa(NumT n)
{
 std::string res;
 return ntoa(n,res);
}

template<typename NumT>
wstring wntoa(NumT n)
{
 std::wstring res;
 return ntoa(n,res);
}

template<typename CharT, typename NumT>
basic_string<CharT>& ntoa(NumT n, int digs, basic_string<CharT> &buf)
{
 basic_stringstream<CharT> stream;
 stream.width(digs);
 stream.fill('0');
 stream<<fixed<<setprecision(digs)<<n;
 return buf=stream.str();
}

template<typename NumT>
string sntoa(NumT n,int digs)
{
 string res;
 return ntoa(n,digs,res);
}

template<typename NumT>
wstring wntoa(NumT n,int digs)
{
 wstring res;
 return ntoa(n,digs,res);
}

// �������, ������������� ����� � ����������������� ������
template<typename CharT, typename NumT>
basic_string<CharT>& ntohex(NumT n, basic_string<CharT> &buf)
{
 basic_stringstream<CharT> stream;
 stream<<hex<<n;
 return buf=stream.str();
}

template<typename CharT, typename NumT>
basic_string<CharT>& ntohex(NumT n, int digs, basic_string<CharT> &buf)
{
 basic_stringstream<CharT> stream;
 stream.width(digs);
 stream.fill('0');
 stream<<hex<<fixed<<setprecision(digs)<<n;
 return buf=stream.str();
}


template<typename NumT>
string sntohex(NumT n)
{
 std::string res;
 return ntohex(n,res);
}

template<typename NumT>
string sntohex(NumT n, int digs)
{
 std::string res;
 return ntohex(n,digs, res);
}

template<typename NumT>
wstring wntohex(NumT n)
{
 std::wstring res;
 return ntohex(n,res);
}

template<typename NumT>
wstring wntohex(NumT n, int digs)
{
 std::wstring res;
 return ntohex(n,digs, res);
}

// �������, ������������� ������ � ������������ �����
template<typename CharT>
double atof(const std::basic_string<CharT> &str)
{
 basic_stringstream<CharT> stream(str);
 double res(0.0);
 stream>>res;
 return res;
}

// �������, ������������� ������ � ����� �����
template<typename CharT>
int atoi(const std::basic_string<CharT> &str)
{
 basic_stringstream<CharT> stream(str);
 int res(0);
 stream>>res;
 return res;
}

// �������, ������������� ����������������� ������ � ����� �����
template<typename CharT, typename T>
T hextoi(const std::basic_string<CharT> &str)
{
 basic_stringstream<CharT> stream(str);
 int res(0);
 stream>>hex>>res;
 return res;
}

// �������� ������� ����� ����� � ��������� digs ������ ����� �������
RDK_LIB_TYPE int fraction(double d, int digs);

// ��������� ������ �� ����������� ����� ��������� 'sep'
// ���������� ����� ���������� �����
template<typename CharT>
int separatestring(const basic_string<CharT> &str, vector<basic_string<CharT> > &output, CharT sep, int num=0, int *lastpos=0)
{
 typename basic_string<CharT>::size_type i=0,j=0;
 int size=0;
 int nnum=(num>0)?num-1:0;

 if(lastpos) *lastpos=0;
 output.resize(0);
 if(str.empty())
  return 0;

 while(i != string::npos && (nnum>=0) )
 {
  i=str.find_first_of(sep,j);
  if(i == j)
  {
   j++;
   continue;
  }

  ++size;
  output.resize(size);
  if(num)
   nnum--;
  if(i == string::npos)
   output[size-1]=str.substr(j);
  else
   output[size-1]=str.substr(j,i-j);
  j=i+1;
  if(j >= str.size())
   break;
 }

 if(lastpos) *lastpos=int(i);

 return size;
}

template<typename CharT>
std::basic_string<CharT> concat_strings(const std::vector<std::basic_string<CharT> > &elements,
                           const std::basic_string<CharT> &separator)
{
 if (!elements.empty())
 {
  std::basic_stringstream<CharT> ss;
  typename std::vector<std::basic_string<CharT> >::const_iterator it = elements.begin();
  while (true)
  {
   ss << *it++;
   if (it != elements.end())
    ss << separator;
   else
    return ss.str();
  }
 }
 return "";
}

/// ���������� ����� � ���� �������� ������ ���� YYYY.MM.DD HH:MM:SS
RDK_LIB_TYPE std::string get_text_time(time_t time_data, char date_sep='.', char time_sep=':');

/// ���������� ����� � ���� �������� ������ ���� YYYYy MMm DDd HHh MMm SS:MSMSs �� ������� � ��������
/// ���������� ������ �� �������� �������, ������� ����������
/// ���� is_full_time==true �� ����� ��������� ��������� ������
RDK_LIB_TYPE std::string get_text_time_from_seconds(double time_data, char date_sep='.', char time_sep=':', bool is_full_time=false);

// ���������� ����� � ���� �������� ������ ���� YYYY/MM/DD HH:MM:SS,MS + ���������� ������ additional_line
//std::string get_text_current_time(char date_sep='/', char time_sep=':', char m_sec_sep=',', std::string additional_line="GMT+04:00");

// ����������� string<->wstring
// !!! ������������-��������� ���������� !!!
// ��������� � http://habrahabr.ru/blogs/cpp/112997/
//@brief ������ ������� ������, ��������� ����������� loc
//   @return ���������� �������� ������ ��� ������ �������� ������, �
//   ������. ���� �������� ������
//RDK_LIB_TYPE std::string& narrow(const std::wstring& wstr, const std::locale& loc, std::string &result);
//RDK_LIB_TYPE std::string& narrow(const std::wstring& wstr, std::string &result, unsigned codepage);

//std::string narrow2(const std::wstring& wstr);

//@brief ��������� ������, ��������� ����������� loc
//   @return ���������� ����������� ������ ��� ������ ����������� ������, �
//   ������, ���� �������� ������.
//RDK_LIB_TYPE std::wstring& widen(const std::string& str, const std::locale& loc, std::wstring &result);
//RDK_LIB_TYPE std::wstring& widen(const std::string& str, std::wstring &result, unsigned codepage);

//std::wstring widen2(const std::string& str);

/// From http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
#if __cplusplus >= 201103L
#include <type_traits>

template <typename T>
inline int signum(T x, std::false_type is_signed)
{
 return T(0) < x;
}

template <typename T>
inline int signum(T x, std::true_type is_signed)
{
 return (T(0) < x) - (x < T(0));
}

template <typename T>
inline int signum(T x)
{
 return signum(x, std::is_signed<T>());
}
#else
template <typename T>
inline int signum(T x)
{
 return (x>0)?1:((x<0)?-1:0);
}
#endif

/// �������� ���������� � ����������� ������� � ������
RDK_LIB_TYPE std::string trim_spaces(const std::string &str);

/// ��������� ���� �� ������� ����� �����
RDK_LIB_TYPE std::string extract_file_path(const std::string& full_name);

/// ��������� ��� ����� �� ������� ����� �����
RDK_LIB_TYPE std::string extract_file_name(const std::string& full_name);

/// ��������� ��� ����� �� ������� ����� ����� �������� ����������
RDK_LIB_TYPE std::string extract_file_name_wo_ext(const std::string& full_name);

/// ��������� ���������� ����� �� ����� �����
RDK_LIB_TYPE std::string extract_file_ext(const std::string& full_name);

/// ������������ ��� ����� ���, ����� ��� ����� ������������� ������������ ��������� ����
RDK_LIB_TYPE string relative_file_path(const string &path, const string &relative_path_base);

/// �������� ��� ��������� ��������� find_str �� ��������� replace_str
RDK_LIB_TYPE string replace_substring(const string &src, const string &find_str, const string &replace_str);



/// ��������� ���� � ������
RDK_LIB_TYPE bool LoadFile(const std::string &file_name, std::string &buffer);
RDK_LIB_TYPE bool LoadFileBin(const std::string &file_name, std::vector<uint8_t> &buffer);

/// ��������� ���� �� ������
RDK_LIB_TYPE bool SaveFile(const std::string &file_name, const std::string &buffer);
RDK_LIB_TYPE bool SaveFileBin(const std::string &file_name, const std::vector<uint8_t> &buffer);

}
#endif



