#ifndef _IO_H_
#define _IO_H_

#include <string>
#include <vector>

using namespace std;

class Reader {
public:
  Reader(void) {}
  Reader(const char *filename);
  // This constructor for use by NewReaderMaybe().  Doesn't call stat().
  Reader(const char *filename, long long int file_size);
  virtual ~Reader(void);
  bool AtEnd(void) const;
  void SeekTo(long long int offset);
  bool ReadInt(int *i);
  int ReadIntOrDie(void);
  bool ReadUnsignedInt(unsigned int *i);
  unsigned int ReadUnsignedIntOrDie(void);
  bool ReadLong(long long int *l);
  long long int ReadLongOrDie(void);
  bool ReadUnsignedLong(unsigned long long int *u);
  unsigned long long int ReadUnsignedLongOrDie(void);
  bool ReadShort(short *s);
  short ReadShortOrDie(void);
  bool ReadChar(char *c);
  char ReadCharOrDie(void);
  bool ReadUnsignedChar(unsigned char *u);
  unsigned char ReadUnsignedCharOrDie(void);
  bool ReadUnsignedShort(unsigned short *u);
  unsigned short ReadUnsignedShortOrDie(void);
  bool ReadDouble(double *d);
  double ReadDoubleOrDie(void);
  bool ReadFloat(float *f);
  float ReadFloatOrDie(void);
  bool ReadReal(float *f);
  bool ReadReal(double *d);
  bool GetLine(string *s);
  bool ReadCString(string *s);
  string ReadCStringOrDie(void);
  // Doesn't do the expected thing for CompressedReader
  long long int BytePos(void) const {return byte_pos_;}
  long long int FileSize(void) const {return file_size_;}
  unsigned char *ReadBytesOrDie(int *num_bytes);
  void ReadEverythingLeft(unsigned char *data);
  int FD(void) const {return fd_;}
  const string &Filename(void) const {return filename_;}

 protected:
  void OpenFile(const char *filename);
  virtual bool Refresh(void);

  static const int kBufSize = 65536;

  int fd_;
  unsigned char *buf_;
  unsigned char *end_read_;
  unsigned char *buf_ptr_;
  int buf_size_;
  long long int file_size_;
  long long int remaining_;
  unsigned char overflow_[100];
  int overflow_size_;
  // Doesn't do the expected thing for CompressedReader
  long long int byte_pos_;
  string filename_;
};

Reader *NewReaderMaybe(const char *filename);

class Writer {
 public:
  Writer(const char *filename);
  Writer(const char *filename, bool modify);
  Writer(const char *filename, int buf_size);
  Writer(const char *filename, bool modify, int buf_size);
  virtual ~Writer(void);

  void WriteInt(int i);
  void WriteUnsignedInt(unsigned int u);
  void WriteLong(long long int l);
  void WriteUnsignedLong(unsigned long long int l);
  void WriteShort(short s);
  void WriteChar(char c);
  void WriteUnsignedChar(unsigned char c);
  void WriteUnsignedShort(unsigned short s);
  void WriteFloat(float f);
  void WriteDouble(double d);
  void WriteReal(float f);
  void WriteReal(double d);
  void WriteCString(const char *s);
  void WriteBytes(unsigned char *bytes, int num_bytes);
  void WriteText(const char *s);
  void SeekTo(long long int offset);
  long long int Tell(void);
  int fd(void) const {return fd_;}
  virtual void Flush(void);

 protected:
  static const int kBufSize = 65536;

  void Init(const char *filename, bool modify, int buf_size);

  int fd_;
  unsigned char *buf_;
  unsigned char *buf_ptr_;
  unsigned char *end_buf_;
  int buf_size_;
  string filename_;
};

class ReadWriter {
public:
  ReadWriter(const char *filename);
  ~ReadWriter(void);
  void SeekTo(long long int offset);
  int ReadIntOrDie(void);
  void WriteInt(int i);
private:
  int fd_;
  string filename_;
};

bool FileExists(const char *filename);
long long int FileSize(const char *filename);
void GetDirectoryListing(const char *dir, vector<string> *listing);
void Mkdir(const char *dir);
void UnlinkFile(const char *filename);
void RemoveFile(const char *filename);
void MoveFile(const char *old_location, const char *new_location);
void CopyFile(const char *old_location, const char *new_location);

#endif
