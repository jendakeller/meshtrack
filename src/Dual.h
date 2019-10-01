#ifndef DUAL_H_
#define DUAL_H_

#include "Vector.h"
#include "SparseVector.h"

template<typename T>
struct Dual
{
public:
  T a;
  SparseVector<T> b;

  Dual();
  Dual(const T& a);
  explicit Dual(const T& a,const SparseVector<T>& b);
  Dual(const Dual<T>& d);

  ~Dual();

  Dual<T>& operator=(const Dual<T>& d);

  Dual<T> operator+=(const Dual<T>& d);
};

template<typename T>
Dual<T>::Dual() {}

template<typename T>
Dual<T>::Dual(const T& a) : a(a) { }

template<typename T>
Dual<T>::Dual(const T& a,const SparseVector<T>& b) : a(a),b(b) { }

template<typename T>
Dual<T>::Dual(const Dual<T>& d) : a(d.a),b(d.b) { }

template<typename T>
Dual<T>& Dual<T>::operator=(const Dual<T>& d)
{
  if (this!=&d)
  {
    a = d.a;
    b = d.b;
  }

  return *this;
}

template<typename T>
Dual<T>::~Dual() { }

template<typename T>
inline Dual<T> operator-(const Dual<T>& x)
{
  return Dual<T>(-x.a,-x.b);
}

template<typename T>
inline Dual<T> operator+(const Dual<T>& x,const Dual<T>& y)
{
  return Dual<T>(x.a+y.a,x.b+y.b);
}

template<typename T>
Dual<T> Dual<T>::operator+=(const Dual<T>& d)
{
  a = a + d.a;
  b = b + d.b;
  return *this;
}

template<typename T>
inline Dual<T> operator+(const T x,const Dual<T>& y)
{
  return Dual<T>(x+y.a,y.b);
}

template<typename T>
inline Dual<T> operator+(const Dual<T>& x,const T y)
{
  return Dual<T>(x.a+y,x.b);
}

template<typename T>
inline Dual<T> operator-(const Dual<T>& x,const Dual<T>& y)
{
  return Dual<T>(x.a-y.a,x.b-y.b);
}

template<typename T>
inline Dual<T> operator-(const T x,const Dual<T>& y)
{
  return Dual<T>(x-y.a,-y.b);
}

template<typename T>
inline Dual<T> operator-(const Dual<T>& x,const T y)
{
  return Dual<T>(x.a-y,x.b);
}

template<typename T>
inline Dual<T> operator*(const Dual<T>& x,const Dual<T>& y)
{
  return Dual<T>(x.a*y.a,x.a*y.b+x.b*y.a);
}

template<typename T>
inline Dual<T> operator*(const T x,const Dual<T>& y)
{
  return Dual<T>(x*y.a,x*y.b);
}

template<typename T>
inline Dual<T> operator*(const Dual<T>& x,const T y)
{
  return Dual<T>(x.a*y,x.b*y);
}

template<typename T>
inline Dual<T> operator/(const Dual<T>& x,const Dual<T>& y)
{
  return Dual<T>(x.a/y.a,(x.b*y.a-x.a*y.b)/(y.a*y.a));
}

template<typename T>
inline bool operator>(const Dual<T>& x,const Dual<T>& y)
{
  return x.a > y.a;
}

template<typename T>
Dual<T> pow(const Dual<T>& x,T a)
{
  Dual<T> temp;
  T deriv,xval,tol;
  xval = x.a;
  tol = 1e-8;
  if (std::abs(xval) < tol)
  {
    if (xval >= 0) { xval = tol; }
    if (xval < 0) { xval = -tol; }
  }
  deriv = a*std::pow(xval,(a-T(1)));
  temp.a = std::pow(x.a,a);
  temp.b = x.b*deriv;
  return temp;
}

template<typename T>
inline Dual<T> operator/(const T x,const Dual<T>& y)
{
  Dual<T> inv = pow(y,T(-1));
  return x*inv;
}

template<typename T>
inline Dual<T> operator/(const Dual<T>& x,const T y)
{
  return Dual<T>(x.a/y,x.b/y);
}

template<typename T>
Dual<T> sqr(const Dual<T>& x)
{
  return Dual<T>(x.a*x.a,T(2)*x.a*x.b);
}

namespace std
{
  template<typename T>
  Dual<T> exp(const Dual<T>& x)
  {
    return Dual<T>(std::exp(x.a),std::exp(x.a)*x.b);
  }

  template<typename T>
  Dual<T> log(const Dual<T>& x)
  {
    return Dual<T>(std::log(x.a),((T(1.0)/x.a))*x.b);
  }

  template<typename T>
  Dual<T> sqrt(const Dual<T>& x)
  {
    return Dual<T>(std::sqrt(x.a),T(0.5)*T(1.0)/(std::sqrt(x.a)+T(0.0000001))*x.b);
  }

  template<typename T>
  Dual<T> abs(const Dual<T>& x)
  {
    return (x.a < 0 ? -x : x);
  }

  template<typename T>
  Dual<T> min(const Dual<T>& x,const T& y)
  {
    return x.a > y ? Dual<T>(y) : x;
  }

  template<typename T>
  Dual<T> sin(const Dual<T>& x)
  {
    return Dual<T>(sin(x.a),x.b*cos(x.a));
  }

  template<typename T>
  Dual<T> cos(const Dual<T>& x)
  {
    return Dual<T>(cos(x.a),x.b*sin(x.a)*(-1.0f));
  }
}


///////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct DenseDual
{
public:
  T a;
  Vector<T> b;

  DenseDual() : a(0) {};
  DenseDual(const DenseDual& d) : a(d.a),b(d.b) { };
  DenseDual& operator=(const DenseDual<T>& d) { a=d.a; b=d.b;}

  DenseDual<T>& operator+=(const Dual<T>& d);
  DenseDual<T>& operator-=(const Dual<T>& d);
  DenseDual<T>& operator+=(const DenseDual<T>& d);
};

template<typename T>
DenseDual<T>& DenseDual<T>::operator-=(const Dual<T>& d)
{
  a -= d.a;

  for(int i=0;i<d.b.n;i++)
  {
    b[d.b.ind[i]] -= d.b.val[i];
  }

  return *this;
}

template<typename T>
DenseDual<T>& DenseDual<T>::operator+=(const Dual<T>& d)
{
  a += d.a;

  for(int i=0;i<d.b.n;i++)
  {
    b[d.b.ind[i]] += d.b.val[i];
  }

  return *this;
}

/*
template<typename T>
DenseDual<T>& DenseDual<T>::operator+=(const DenseDual<T>& d)
{
  a += d.a;

  for(int i=0;i<d.b.size();i++)
  {
    b[i] += d.b[i];
  }

  return *this;
}
*/
#endif
