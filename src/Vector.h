#ifndef VECTOR_H_
#define VECTOR_H_

#include <cassert>

template<typename T>
class Vector
{
public:  
  Vector();
  explicit Vector(int size);
  Vector(const Vector<T>& a);
  ~Vector();

  Vector& operator=(const Vector<T>& a);      
  
  inline T&       operator[](int i);
  inline const T& operator[](int i) const;  
  inline T&       operator()(int i);  
  inline const T& operator()(int i) const;
   
  int      size() const;
  T*       data();  
  const T* data() const;
     
private:  
  int n;
  T* v;
};

template<typename T> Vector<T> operator-(const Vector<T>& x);
template<typename T> Vector<T> operator+(const Vector<T>& x,const Vector<T>& y);
template<typename T> Vector<T> operator-(const Vector<T>& x,const Vector<T>& y);
template<typename T> Vector<T> operator*(const T x,const Vector<T>& y);
template<typename T> Vector<T> operator*(const Vector<T>& x,const T y);
template<typename T> Vector<T> operator/(const Vector<T>& x,const T y);

template<typename T> T dot(const Vector<T>& x,const Vector<T>& y);
template<typename T> T mag2(const Vector<T>& x);

template<typename T>
Vector<T>::Vector() : n(0),v(0) {}

template<typename T>
Vector<T>::Vector(int size)
{
  assert(size>=0);
  n = size;
  v = n>0 ? new T[n] : 0;
}
   
template<typename T>
Vector<T>::Vector(const Vector<T>& u)
{    
  n = u.n;
  
  if (n>0)
  {
    v = new T[n];
    for(int i=0;i<n;i++) v[i] = u.v[i];
  }
  else
  {
    v = 0;
  }
}

template<typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& u)
{
  if (this!=&u)
  {
    if (n==u.n)
    {
      for(int i=0;i<n;i++) v[i] = u.v[i];
    }
    else
    {
      delete[] v;
      n = u.n;
  
      if (n>0)
      {
        v = new T[n];  
        for(int i=0;i<n;i++) v[i] = u.v[i];
      }
      else
      {
        v = 0;
      }
    }
  }
  else
  {
  }    
  
  return *this;
}

template<typename T>
Vector<T>::~Vector()
{
  delete[] v;
}

template<typename T>  
T& Vector<T>::operator[](int i)
{
  assert(i>=0 && i<n);
  return v[i];
}

template<typename T>
const T& Vector<T>::operator[](int i) const
{
  assert(i>=0 && i<n);
  return v[i];
}

template<typename T>  
T& Vector<T>::operator()(int i)
{
  assert(i>=0 && i<n);
  return v[i];
}

template<typename T>
const T& Vector<T>::operator()(int i) const
{
  assert(i>=0 && i<n);
  return v[i];
}

template<typename T>  
int Vector<T>::size() const
{
  return n;
}
  
template<typename T>  
T* Vector<T>::data()
{
  return v;
}
  
template<typename T>  
const T* Vector<T>::data() const
{
  return v;
}

template<typename T>
Vector<T> operator-(const Vector<T>& x)
{
  Vector<T> z(x.size());
  for(int i=0;i<z.size();i++) { z(i) = -x(i); }
  return z;
}

template<typename T>
Vector<T> operator+(const Vector<T>& x,const Vector<T>& y)
{
  assert((x.size()==y.size()) || x.size()==0 || y.size()==0);
  
  if      (y.size()==0) { return x; }
  else if (x.size()==0) { return y; }

  Vector<T> z(x.size());
  for(int i=0;i<z.size();i++) { z(i) = x(i) + y(i); }
  return z;
}

template<typename T>
Vector<T> operator-(const Vector<T>& x,const Vector<T>& y)
{
  assert((x.size()==y.size()) || x.size()==0 || y.size()==0);

  if      (y.size()==0) { return  x; }
  else if (x.size()==0) { return -y; }
 
  Vector<T> z(x.size());
  for(int i=0;i<z.size();i++) { z(i) = x(i) - y(i); }
  return z;
}

template<typename T>
Vector<T> operator*(const T x,const Vector<T>& y)
{
  Vector<T> z(y.size());
  for(int i=0;i<z.size();i++) { z(i) = x * y(i); }
  return z;
}

template<typename T>
Vector<T> operator*(const Vector<T>& x,const T y)
{
  Vector<T> z(x.size());
  for(int i=0;i<z.size();i++) { z(i) = x(i) * y; }
  return z;
}

template<typename T>
Vector<T> operator/(const Vector<T>& x,const T y)
{
  Vector<T> z(x.size());
  for(int i=0;i<z.size();i++) { z(i) = x(i) / y; }
  return z;
}

template<typename T>
T dot(const Vector<T>& x,const Vector<T>& y)
{
  assert((x.size()==y.size()) || x.size()==0 || y.size()==0);

  if (x.size()==0 || y.size()==0) { return T(0); }
  
  T sum = T(0);
  for(int i=0;i<x.size();i++) { sum += x(i) * y(i); }
  return sum;
}

template<typename T> T
mag2(const Vector<T>& x)
{
  T sum = T(0);
  for(int i=0;i<x.size();i++) { sum += x(i)*x(i); }
  return sum;
}

#endif
