#include <cfloat>
#include <climits>
#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <direct.h>

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/SVD>
#include <Eigen/Geometry>

#include "glew.h"

#include "jzq.h"
#include "jzq_gl.h"
#include "timer.h"
#include "imageio.h"
#include "video.h"
#include "gui.h"

#include "Vector.h"
#include "Dual.h"
#include "lbfgs.h"

#include "skinningModel.h"
#include "animation.h"

#include "utils_sampling.hpp"

#define NANORT_IMPLEMENTATION
#include "nanort.h"

#define ID __COUNTER__
#define USE_OPENGL 1
//#define DEBUG_BLOBS 1

enum ModelViewMode
{
  MODEL_VIEW_MODE_NORMALS,
  MODEL_VIEW_MODE_COLORS,
  MODEL_VIEW_MODE_WEIGHTS
};

enum Mode
{
  ModeNone,
  ModeZoomIn,
  ModeZoomOut,
  ModePan,
  ModeRotate,
  ModePlacePoint,
  ModeDragPoint,
  ModePick,
  ModeFix
};

struct ImageBlob
{
  V2f mu;
  float sigma;
  V3f rgbColor;
  V3f hsvColor;

  ImageBlob() { }
  ImageBlob(const V2f& mu, float sigma, const V3f& rgbColor, const V3f& hsvColor) :
  mu(mu), sigma(sigma), rgbColor(rgbColor), hsvColor(hsvColor) { }
};

struct Camera
{
  Mat3x4f P;

  Mat3x3f K;
  Mat3x3f R;
  Vec3f   C;

  Camera() { }
  Camera(const Mat3x4f& P) : P(P) { }
};

struct View
{
  Camera camera;
  A2V3uc image;
  A2uc   mask;
  
  std::vector<ImageBlob> imageBlobs;
  float sumBlobsOverlaps;
};

struct View2D
{
  float width;    // width of the viewport rectangle (in screen pixels)
  float height;   // height of the viewport rectangle (in screen pixels)
  float panX;     // x-coordinate of the viewport center relative to canvas origin (in canvas pixels)
  float panY;     // y-coordinate of the viewport center relative to canvas origin (in canvas pixels)
  float zoom;     // size of the canvas pixel (in screen pixels) as seen through viewport
  float angle;    // rotation of the viewport rectangle (around its center) relative to canvas basis (in radians)
};

struct Ray
{
  const Vec<3,float> o;
  const Vec<3,float> d;
  const Vec<3,float> invd;

  inline Ray(const Vec<3,float>& o,
             const Vec<3,float>& d) : o(o),d(d),invd(1.0f/d(0),1.0f/d(1),1.0f/d(2)) { }
};

struct Anchor
{
  V3f meshPoint;
  V2f viewPoint;
  std::vector<float> jointWeights;
  int viewId;
};

struct AABB
{
  V2f bboxMin;
  V2f bboxMax;
};

struct BBoxSelector
{
  V2f bboxMin;
  V2f bboxMax;
  int viewId;
};

struct MeshPointSamples
{
  std::vector<V3f> points;
  std::vector<V3i> triangles; // source triangles that points are taken from
  std::vector<V3f> barycentrics; // barycentric coordinates of points inside the triangles
  A2f weights; // skinning weights of the points

  A2f initialVisibilities;
  std::vector<int> initialVisibilitiesArgMax;
};

//======================== SHORT DUAL NUMBERS =========================

template<int N>
struct ShortDual
{
  float a;
  Vec<N,float> b;
  inline ShortDual() {}

  inline ShortDual(const float &xa,const Vec<N,float> &xb) : a(xa),b(xb) {}
};

template<int N>
inline ShortDual<N> operator+(const ShortDual<N> &x,const ShortDual<N> &y)
{
  return ShortDual<N>(x.a+y.a,x.b+y.b);
}

template<int N>
inline ShortDual<N> operator+(const float x,const ShortDual<N> &y)
{
  return ShortDual<N>(x+y.a,y.b);
}

template<int N>
inline ShortDual<N> operator+(const ShortDual<N> &x,const float y)
{
  return ShortDual<N>(x.a+y,x.b);
}

template<int N>
inline ShortDual<N> operator-(const ShortDual<N> &x)
{
  return ShortDual<N>(-x.a,-x.b);
}

template<int N>
inline ShortDual<N> operator-(const ShortDual<N> &x,const ShortDual<N> &y)
{
  return ShortDual<N>(x.a-y.a,x.b-y.b);
}

template<int N>
inline ShortDual<N> operator-(const float x,const ShortDual<N> &y)
{
  return ShortDual<N>(x-y.a,y.b*float(-1));
}

template<int N>
inline ShortDual<N> operator-(const ShortDual<N> &x,const float y)
{
  return ShortDual<N>(x.a-y,x.b);
}

template<int N>
inline ShortDual<N> operator*(const ShortDual<N> &x,const ShortDual<N> &y)
{
  return ShortDual<N>(x.a*y.a,x.a*y.b+x.b*y.a);
}

template<int N>
inline ShortDual<N> operator*(const float x,const ShortDual<N> &y)
{
  return ShortDual<N>(x*y.a,x*y.b);
}

template<int N>
inline ShortDual<N> operator*(const ShortDual<N> &x,const float y)
{
  return ShortDual<N>(x.a*y,x.b*y);
}

template<int N>
inline ShortDual<N> operator/(const ShortDual<N> &x,const ShortDual<N> &y)
{
  return ShortDual<N>(x.a/y.a,(x.b*y.a-x.a*y.b)/(y.a*y.a));
}

template<int N>
ShortDual<N> pow (const ShortDual<N>& x,float a)
{
  ShortDual<N> temp;
  float deriv,xval,tol;
  xval = x.a;
  tol = 1e-8;
  if (fabs(xval) < tol)
  {
    if (xval >= 0)
      xval = tol;
    if (xval < 0)
      xval = -tol;
  }
  deriv = a*pow(xval,(a-1));
  temp.a = pow(x.a,a);  //Use actual x value, only use tol for derivs
  temp.b = x.b*deriv;
  return temp;
}

template<int N>
inline ShortDual<N> operator/(const float x,const ShortDual<N> &y)
{
  ShortDual<N> inv = pow(y,-1);
  return x*inv;
}


template<int N>
inline ShortDual<N> operator/(const ShortDual<N> &x,const float y)
{
  return ShortDual<N>(x.a/y,x.b/y);
}

template<int N>
ShortDual<N> sqr(const ShortDual<N> &x)
{
  return ShortDual<N>(x.a*x.a,float(2)*x.a*x.b);
}

template<int N>
ShortDual<N> exp(const ShortDual<N> &x)
{
  return ShortDual<N>(std::exp(x.a),std::exp(x.a)*x.b);
}


template<int N>
ShortDual<N> sqrt(const ShortDual<N>& x)
{
  return ShortDual<N>(std::sqrt(x.a),0.5f*(1.0f)/(std::sqrt(x.a)+0.00001f)*x.b);
}

template<int N>
ShortDual<N> abs(const ShortDual<N>& x)
{
  return (x.a < 0.0f ? -1.0f*x : x);
}

//=================== END OF SHORT DUAL NUMBERS =================

struct Gauss
{
  V3f mu;
  float sigma;
};

struct BlobModel
{
  std::vector<Gauss> blobs;
  std::vector<V3f> colors;
  std::vector<V3f> hsvColors;
  A2f blobWeights;

  void updateBlobs(const std::vector<V3f>& vertices, const A2f& weights, std::vector<Gauss>& blobs, int jointId);
  void placeBlobsOnBone(const SkinningModel& model, const std::vector<V3f>& joints, int childJointId, std::vector<Gauss>& subBlobs);
  std::vector<Gauss> kMeans(const std::vector<V3f>& vertices, const A2f& weights, const std::vector<V3f>& pivots, int numClusters, int jointId, int iters, bool fixedClusters=false);
  std::vector<V3f> deformBlobCenters(const SkinningModel& model, const TRSA<float>& trsa) const;
  std::vector<AABB> getBBoxes(const TRSA<float>& trsa, const SkinningModel& model, const std::vector<View>& views) const;
  void updateBlobColors(const std::vector<View>& views, const SkinningModel& model, const TRSA<float>& trsa);
  void updateWeights(const std::vector<V3f>& vertices, const A2f& weights);
  void build(const SkinningModel& model);
};


template<typename T>
Mat<3,3,T> rotationMatrix(const Vec<3,T>& angles)
{
  T cx = std::cos(angles(0));
  T cy = std::cos(angles(1));
  T cz = std::cos(angles(2));
  T sx = std::sin(angles(0));
  T sy = std::sin(angles(1));
  T sz = std::sin(angles(2));

  // Rz*Rx*Ry
  return Mat<3,3,T>(-sx*sy*sz+cy*cz,  sx*sy*cz+cy*sz, -cx*sy,
                             -cx*sz,           cx*cz,     sx,
                     sx*cy*sz+sy*cz, -sx*cy*cz+sy*sz,  cx*cy);
  // Rz*Ry*Rx
  // return Mat<3,3,T>(cy*cz, cz*sx*sy-cx*sz, cx*cz*sy+sx*sz,
  //                   cy*sz, sx*sy*sz+cx*cz, cx*sy*sz-cz*sx,
  //                     -sy,          cy*sx,          cx*cy);
}

template<typename T>
Mat<4,4,T> trsMatrix(const TRS<T>& trs)
{
  Mat<3,3,T> R = rotationMatrix(trs.r);

  const Vec<3,T> t = trs.t;

  const T s = trs.s;
  
  return Mat<4,4,T>(s*R(0,0),s*R(0,1),s*R(0,2),t(0),
                    s*R(1,0),s*R(1,1),s*R(1,2),t(1),
                    s*R(2,0),s*R(2,1),s*R(2,2),t(2),
                           0,       0,       0,  1);
}

template<typename T>
Vec<3,T> e2p(const Vec<2,T>& x) { return Vec<3,T>(x(0),x(1),T(1)); }

template<typename T>
Vec<2,T> p2e(const Vec<3,T>& x) { return Vec<2,T>(x(0)/x(2),x(1)/x(2)); }

template<typename T>
Vec<3,T> p2e(const Vec<4,T>& x) { return Vec<3,T>(x(0)/x(3),x(1)/x(3),x(2)/x(3)); }

template<typename T>
Vec<4,T> e2p(const Vec<3,T>& x) { return Vec<4,T>(x(0),x(1),x(2),T(1)); }

V3f unproject(const Camera& camera,const V2f& x)
{
  return transpose(camera.R)*(inverse(camera.K)*e2p(x));
}

template<typename T>
const Mat<4,4,T> evalJointTransform(const int jointId,
                                    const std::vector<Joint<float>> joints,
                                    const std::vector<Vec<3,T>> angles)
{
  Mat<3,3,T> R = rotationMatrix(angles[jointId]);

  const Vec<3,T> t = Vec<3,T>(joints[jointId].offset);

  Mat<4,4,T> Rt = Mat<4,4,T>(R(0,0),R(0,1),R(0,2),t(0),
                             R(1,0),R(1,1),R(1,2),t(1),
                             R(2,0),R(2,1),R(2,2),t(2),
                             0,     0,     0,     1);
  
  const int parentId = joints[jointId].parentId;
  if (parentId>=0)
  {
    return evalJointTransform(parentId, joints, angles)*Rt;
  }
  else
  {
    return Rt;
  }
}

template<typename T>
std::vector<Vec<3,T>> deformVertices(const std::vector<Vec<3,T>>& orgVertices,
                                     const A2f& vertexJointWeights,
                                     const std::vector<Joint<T>>& orgJoints,
                                     const std::vector<Vec<3,T>>& orgAngles,
                                     const std::vector<Joint<T>>& curJoints,
                                     const std::vector<Vec<3,T>>& curAngles)
{
  std::vector<Vec<3,T>> deformedVertices(orgVertices.size());

  std::vector<Mat<4,4,T>> Ms(orgJoints.size());

  #pragma omp parallel for
  for (int j=0;j<orgJoints.size();j++)
  {
    const Mat<4,4,T> M0 = evalJointTransform(j,orgJoints,orgAngles);
    const Mat<4,4,T> M1 = evalJointTransform(j,curJoints,curAngles);

    Ms[j] = M1*inverse(M0);
  }

  #pragma omp parallel for
  for (int i=0;i<orgVertices.size();i++)
  {
    const Vec<3,T> ov = orgVertices[i];
    Vec<3,T> dv = Vec<3,T>(0,0,0);
    for (int j=0;j<orgJoints.size();j++)
    {
      Vec<3,T> dvPart = p2e(Ms[j]*e2p(ov));
      float w = vertexJointWeights(j,i);
      dv += Vec<3,T>(w*dvPart(0), w*dvPart(1), w*dvPart(2));
    }
    deformedVertices[i] = dv;
  }

  return deformedVertices;
}

void deformSamplePoints(const std::vector<V3f>& samplePoints,
                        const std::vector<Mat<4,4,Dual<float>>>& Ms,
                        const A2f& sampleWeights,
                        std::vector<Vec<3,Dual<float>>>* out_deformedSamplePoints)
{
  std::vector<Vec<3,Dual<float>>>& deformedSamplePoints = *out_deformedSamplePoints;

  int numVars = 0;
  for (int m=0;m<Ms.size();m++)
  {
    for (int i=0;i<4;i++)
    for (int j=0;j<4;j++)
    {
      for (int k=0;k<Ms[m](i,j).b.n;k++)
      {
        numVars = std::max(numVars,Ms[m](i,j).b.ind[k]+1);
      }
    }
  }

  std::vector<Vec<3,std::vector<float>>> perthread_bs(omp_get_max_threads());
  for (int t=0;t<perthread_bs.size();t++)
  {
    for (int i=0;i<3;i++) { perthread_bs[t][i] = std::vector<float>(numVars); }
  }

  #pragma omp parallel for
  for (int s=0;s<samplePoints.size();s++)
  {
    const V3f sp = samplePoints[s];

    V3f a = V3f(0,0,0);
    Vec<3,std::vector<float>>& b = perthread_bs[omp_get_thread_num()];
    for (int i=0;i<3;i++) { for (int j=0;j<b[i].size();j++) { b[i][j] = 0; } }

    for (int m=0;m<Ms.size();m++)
    {
      const float weight = sampleWeights(m,s);

      if (weight>0)
      {
        const Mat<4,4,Dual<float>>& M = Ms[m];

        for (int i=0;i<3;i++)
        {
          for (int j=0;j<3;j++)
          {
            const Dual<float>& Mij = M(i,j);
            a[i] += weight*Mij.a*sp[j];
            for (int k=0;k<Mij.b.n;k++) { b[i][Mij.b.ind[k]] += weight*Mij.b.val[k]*sp[j]; }
          }
          
          const Dual<float>& Mij = M(i,3);
          a[i] += weight*Mij.a;
          for (int k=0;k<Mij.b.n;k++) { b[i][Mij.b.ind[k]] += weight*Mij.b.val[k]; }
        }
      }
    }

    Vec<3,Dual<float>>& dsp = deformedSamplePoints[s];
    for (int i=0;i<3;i++)
    {
      dsp[i].a = a[i];
      dsp[i].b.n = 0;
      dsp[i].b.ind->clear();
      dsp[i].b.val->clear();
      for (int j=0;j<b[i].size();j++)
      {
        if (std::abs(b[i][j])>0)
        {
          dsp[i].b.n++;
          dsp[i].b.ind->push_back(j);
          dsp[i].b.val->push_back(b[i][j]);
        }
      }
    }
  }
}

//======================== INTERSECTIONS ========================

inline bool solveQuadratic(float A,float B,float C,float *t0,float *t1)
{
  float discrim = B * B - 4.f * A * C;
  if (discrim < 0.) return false;
  float rootDiscrim = sqrtf(discrim);

  float q;
  if (B < 0) q = -.5f * (B - rootDiscrim);
  else       q = -.5f * (B + rootDiscrim);
  *t0 = q / A;
  *t1 = C / q;
  if (*t0 > *t1) { std::swap(*t0, *t1); }
  return true;
}

bool intersectSphere(const Ray& r, const V3f& center, float radius, float* tHit)
{
  const Ray ray = Ray(r.o-center,r.d);

  float A = ray.d(0)*ray.d(0) + ray.d(1)*ray.d(1) + ray.d(2)*ray.d(2);
  float B = 2 * (ray.d(0)*ray.o(0) + ray.d(1)*ray.o(1) + ray.d(2)*ray.o(2));
  float C = ray.o(0)*ray.o(0) + ray.o(1)*ray.o(1) +
            ray.o(2)*ray.o(2) - radius*radius;

  float t0, t1;
  if (!solveQuadratic(A, B, C, &t0, &t1))
      return false;

  if (t1 < 0) return false;
  float thit = t0;
  if (t0 < 0)
  {
    thit = t1;
  }

  *tHit = thit;

  return true;
}

bool intersectTriangle(const Ray& ray,const float t0,const float t1,const V3f& v0,const V3f v1,const V3f& v2,float* out_t)
{
  const Vec3f e01 = v1-v0;
  const Vec3f e02 = v2-v0;

  const Vec3f s1 = cross(ray.d,e02);
  const float denom = dot(s1,e01);

  if (denom==0.0f) return false;
  const float invDenom = 1.f/denom;

  const Vec3f d = ray.o-v0;
  const float b1 = dot(d,s1)*invDenom;
  if (b1<0.0f || b1>1.0f) return false;

  const Vec3f s2 = cross(d,e01);
  const float b2 = dot(ray.d, s2)*invDenom;
  if (b2<0.0f || b1+b2>1.0f) return false;

  const float t = dot(e02,s2)*invDenom;
  if (t<t0 || t>t1) return false;

  *out_t = t;

  return true;
}

bool intersectMesh(const Ray& ray,
                   const std::vector<V3f>& vertices,
                   const std::vector<V3i>& triangles,                   
                   int* out_id,
                   float* out_t)
{
  float tnear = FLT_MAX;
  int& id = *out_id;

  bool hit = false;

  
  for (int i=0;i<triangles.size();i++)
  {
    const V3i triangle = triangles[i];

    const V3f& v0 = vertices[triangle[0]];
    const V3f& v1 = vertices[triangle[1]];
    const V3f& v2 = vertices[triangle[2]];

    if (intersectTriangle(ray,0.000001f,tnear,v0,v1,v2,&tnear))
    {
      id = i;
      hit = true;
    }
  }

  *out_t = tnear;

  return hit;
}

// =================== END OF INTERSECTIONS ========================

void rgb2hsv(float r,float g,float b,float* h,float* s,float* v)
{
  const float rgbMin = std::min(r,std::min(g,b));
  const float rgbMax = std::max(r,std::max(g,b));

  const float delta = rgbMax - rgbMin;

  *v = rgbMax;

  *s = (rgbMax > 0) ? (delta / rgbMax) : 0;

  if (delta > 0)
  {
    if (r>=g && r>=b) { *h = (       (g-b) / delta) * 60.0f / 360.0f; }
    if (g>=r && g>=b) { *h = (2.0f + (b-r) / delta) * 60.0f / 360.0f; }
    if (b>=r && b>=g) { *h = (4.0f + (r-g) / delta) * 60.0f / 360.0f; }

    if (*h<0.0f) *h = 1.0f+*h;
  }
  else
  {
    *h = 0;
  }
}

V3f rgb2hsv(const V3f& c)
{
  const float r = c(0);
  const float g = c(1);
  const float b = c(2);

  float h,s,v;

  rgb2hsv(r,g,b,&h,&s,&v);

  return V3f(h,s,v);
}

V3f hsvCone(const V3f hsv)
{
  const float h = hsv(0);
  const float s = hsv(1);
  const float v = hsv(2);

  return V3f(cos(6.28318530718f*h),v,sin(6.28318530718f*h));
}

//=========================== BLOB MODEL ========================

void BlobModel::updateBlobs(const std::vector<V3f>& vertices, const A2f& weights, std::vector<Gauss>& blobs, int jointId)
{
  std::vector<float> sumW(blobs.size(), 0.0f);
  std::vector<V3f> mus(blobs.size(), V3f(0,0,0));

  for (int i=0;i<blobs.size();i++)
  {
    blobs[i].sigma = 0.0f;
  }

  for (int i=0;i<vertices.size();i++)
  {
    float w = weights(jointId, i);
    if (w > 0.5f)
    {
      w = 1.0f;
      float minD2 = +FLT_MAX;
      int bId = -1;
      for (int j=0;j<blobs.size();j++)
      {
        V3f diff = vertices[i] - blobs[j].mu;
        float d2 = dot(diff,diff);
        if (d2 < minD2)
        {
          minD2 = d2;
          bId = j;
        }
      }
      mus[bId] += w*vertices[i];
      sumW[bId] += w;
    }
  }

  for (int i=0;i<blobs.size();i++)
  {
    mus[i] = mus[i] / sumW[i];
  }

  for (int i=0;i<vertices.size();i++)
  {
    float w = weights(jointId, i);
    if (w > 0.5f)
    {
      w = 1.0f;
      float minD2 = +FLT_MAX;
      int bId = -1;
      for (int j=0;j<blobs.size();j++)
      {
        V3f diff = vertices[i] - blobs[j].mu;
        float d2 = dot(diff,diff);
        if (d2 < minD2)
        {
          minD2 = d2;
          bId = j;
        }
      }
      blobs[bId].sigma += w*minD2;
    }
  }

  for (int i=0;i<blobs.size();i++)
  {
    blobs[i].mu = mus[i];
    blobs[i].sigma = sqrt(blobs[i].sigma / sumW[i]);
  }
}

void BlobModel::placeBlobsOnBone(const SkinningModel& model, const std::vector<V3f>& joints, int childJointId, std::vector<Gauss>& subBlobs)
{
  int jointId = model.joints[childJointId].parentId;

  for (int i=0;i<subBlobs.size();i++)
  {
    float alpha = (2*i+1)/float(2*subBlobs.size());
    subBlobs[i].mu = alpha*joints[jointId] + (1.0f-alpha)*joints[childJointId];
  }
  updateBlobs(model.vertices, model.weights, subBlobs, jointId);
}

std::vector<Gauss> BlobModel::kMeans(const std::vector<V3f>& vertices, const A2f& weights, const std::vector<V3f>& pivots, int numClusters, int jointId, int iters, bool fixedClusters)
{
  std::vector<Gauss> clusters(numClusters);
  std::vector<Gauss> nextClusters(numClusters);
  std::vector<float> sumW(numClusters);

  // cluster center initialization
  if (pivots.size())
  {
    for (int i=0;i<numClusters;i++)
    {
      clusters[i].mu = pivots[i];
    }
  }
  else
  {
    for (int i=0;i<numClusters;i++)
    {
      int vId;
      while (true)
      {
        int r = rand()*(RAND_MAX+1)+rand(); // Hack for increasing RAND_MAX
        vId = r % vertices.size();
        if (weights(jointId, vId) > 0.5f)
        {
          break;
        }
      }
      clusters[i].mu = vertices[vId];
    }
  }

  for (int iter=0;iter<iters;iter++)
  {
    for (int i=0;i<numClusters;i++)
    {
      nextClusters[i].mu = V3f(0,0,0);
      nextClusters[i].sigma = 0.0f;
      sumW[i] = 0.0f;
    }

    for (int i=0;i<vertices.size();i++)
    {
      float w = weights(jointId, i);
      if (w > 0.5f)
      {
        w = 1.0f;
        float minD2 = +FLT_MAX;
        int cId = -1;
        for (int j=0;j<numClusters;j++)
        {
          V3f diff = vertices[i] - clusters[j].mu;
          float d2 = dot(diff,diff);
          if (d2 < minD2)
          {
            minD2 = d2;
            cId = j;
          }
        }
        nextClusters[cId].mu += w*vertices[i];
        nextClusters[cId].sigma += w*minD2;
        sumW[cId] += w;
      }
    }

    for (int i=0;i<numClusters;i++)
    {
      nextClusters[i].mu = nextClusters[i].mu / sumW[i];
      nextClusters[i].sigma /= sumW[i];
    }

    if (!fixedClusters)
    {
      clusters = nextClusters;
    }
  }

  for (int i=0;i<numClusters;i++)
  {
    clusters[i].sigma = sqrt(fixedClusters ? nextClusters[i].sigma : clusters[i].sigma);
  }

  return clusters;
}

std::vector<V3f> BlobModel::deformBlobCenters(const SkinningModel& model, const TRSA<float>& trsa) const
{
  std::vector<V3f> blobCenters(blobs.size());
  for (int i=0;i<blobCenters.size();i++)
  {
    blobCenters[i] = blobs[i].mu;
  }
  
  const Mat4x4f M = trsMatrix(trsa.trs);
  
  std::vector<V3f> orgAngles(trsa.angles.size(), V3f(0,0,0));
  blobCenters = deformVertices(blobCenters,
                               blobWeights,
                               model.joints,
                               orgAngles,
                               model.joints,
                               trsa.angles);

  for (int i=0;i<blobCenters.size();i++)
  {
    blobCenters[i] = p2e(M*e2p(blobCenters[i]));
  }

  return blobCenters;
}

std::vector<AABB> BlobModel::getBBoxes(const TRSA<float>& trsa, const SkinningModel& model, const std::vector<View>& views) const
{
  std::vector<AABB> bboxes(views.size());
  std::vector<V3f> blobCenters = deformBlobCenters(model, trsa);

  #pragma omp parallel for
  for (int i=0;i<views.size();i++)
  {
    const View& view = views[i];
    const A2V3uc& I = view.image;

    AABB& aabb = bboxes[i];
    aabb.bboxMin = V2f(I.width(), I.height());
    aabb.bboxMax = V2f(0, 0);
    float f = (fabs(view.camera.K(0,0)) + fabs(view.camera.K(1,1))) / 2.0f;

    for (int j=0;j<blobCenters.size();j++)
    {
      V3f xyw = view.camera.P*e2p(blobCenters[j]);
      V2f xy = p2e(xyw);
      V2f sigma2D(blobs[j].sigma * f / xyw[2], blobs[j].sigma * f / xyw[2]);
      
      aabb.bboxMin = std::min(xy - sigma2D, aabb.bboxMin);
      aabb.bboxMax = std::max(xy + sigma2D, aabb.bboxMax);
    }
    aabb.bboxMin[0] = std::max(aabb.bboxMin[0], 0.0f);
    aabb.bboxMin[1] = std::max(aabb.bboxMin[1], 0.0f);
    aabb.bboxMax[0] = std::min(aabb.bboxMax[0], I.width() -1.0f);
    aabb.bboxMax[1] = std::min(aabb.bboxMax[1], I.height()-1.0f);

    float w = aabb.bboxMax[0] - aabb.bboxMin[0];
    float h = aabb.bboxMax[1] - aabb.bboxMin[1];

    aabb.bboxMin[0] = std::max(aabb.bboxMin[0] - 0.1f*w, 0.0f);
    aabb.bboxMin[1] = std::max(aabb.bboxMin[1] - 0.1f*h, 0.0f);
    aabb.bboxMax[0] = std::min(aabb.bboxMax[0] + 0.1f*w, I.width() -1.0f);
    aabb.bboxMax[1] = std::min(aabb.bboxMax[1] + 0.1f*h, I.height()-1.0f);
  }

  return bboxes;
}

void BlobModel::updateBlobColors(const std::vector<View>& views, const SkinningModel& model, const TRSA<float>& trsa)
{
  std::vector<V3f> blobCenters = deformBlobCenters(model, trsa);

  std::vector<int> counts(colors.size(), 0);
  for (int i=0;i<colors.size();i++)
  {
    colors[i] = V3f(0,0,0);
  }

  for (int i=0;i<views.size();i++)
  {
    const View& view = views[i];
    const A2V3uc& image = view.image;

    Mat3x3f Pinv = transpose(view.camera.R)*inverse(view.camera.K);
    
    // Compute model's axis aligned bounding box in the current view
    V2f bboxMin(image.width(),image.height());
    V2f bboxMax(0,0);

    float f = (fabs(view.camera.K(0,0)) + fabs(view.camera.K(1,1))) / 2.0f;
    for (int i=0;i<blobs.size();i++)
    {
      V3f xyw = view.camera.P*e2p(blobCenters[i]);
      V2f xy = p2e(xyw);
      V2f sigma2D(blobs[i].sigma * f / xyw[2], blobs[i].sigma * f / xyw[2]);
      
      bboxMin = std::min(xy - sigma2D, bboxMin);
      bboxMax = std::max(xy + sigma2D, bboxMax);
    }

    int x0 = (int)bboxMin[0];
    int y0 = (int)bboxMin[1];
    int x1 = (int)bboxMax[0];
    int y1 = (int)bboxMax[1];

    for (int y=y0;y<y1;y++)
    for (int x=x0;x<x1;x++)
    {
      float tHit = FLT_MAX;
      V3f d = normalize(Pinv*e2p(V2f(x,y)));

      for (int i=0;i<blobs.size();i++)
      {
        if (intersectSphere(Ray(view.camera.C,d), blobCenters[i], blobs[i].sigma, &tHit))
        {
          colors[i] += V3f(image(x,y));
          counts[i]++;
        }
      }
    }
  }

  for (int i=0;i<colors.size();i++)
  {
    colors[i] = (counts[i]>0) ? colors[i] / (255.0f * counts[i]) : V3f(1,0,0);
    hsvColors[i] = hsvCone(rgb2hsv(colors[i]));
  }
}

void BlobModel::updateWeights(const std::vector<V3f>& vertices, const A2f& weights)
{
  for (int i=0;i<blobWeights.numel();i++)
  {
    blobWeights[i] = 0.0f;
  }

  for (int i=0;i<vertices.size();i++)
  {
    const V3f& v = vertices[i];
    int bId = -1;
    float minDist = +FLT_MAX;

    for (int j=0;j<blobs.size();j++)
    {
      float d = norm(blobs[j].mu - v) / blobs[j].sigma;
      if (d < minDist)
      {
        minDist = d;
        bId = j;
      }
    }

    for (int j=0;j<blobWeights.width();j++)
    {
      blobWeights(j,bId) += weights(j,i);
    }
  }

  for (int y=0;y<blobWeights.height();y++)
  {
    float s = 0.0f;
    for (int x=0;x<blobWeights.width();x++)
    {
      s += blobWeights(x,y);
    }
    for (int x=0;x<blobWeights.width();x++)
    {
      blobWeights(x,y) /= s;
    }
  }
}

void BlobModel::build(const SkinningModel& model)
{
  const std::vector<V3f>& joints = model.getRestPoseJointPositions();

  std::vector<V3f> pivots(2);
  std::vector<Gauss> subBlobs;

  pivots[0] = joints[1];
  pivots[1] = joints[2];
  
  float dsFac = 1.0f;

  // kostrc
  //subBlobs = kMeans(model.vertices, model.weights, pivots, 2, 0, 100);
  //kMeans(vertices, weights, pivots, numClusters, jointId, iters)
  std::vector<V3f> rootPivots(3);
  rootPivots[2] = 0.5f*(joints[1]+joints[2]);
  rootPivots[0] = rootPivots[2] + 1.0f*(joints[1]-rootPivots[2]);
  rootPivots[1] = rootPivots[2] + 1.0f*(joints[2]-rootPivots[2]);
  
  // subBlobs = kMeans(model.vertices, model.weights, rootPivots, 3, 0, 1, true);
  // for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma *= 0.6f; }
  // for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma /= dsFac; }
  // blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // pupek
  // subBlobs = kMeans(model.vertices, model.weights, pivots, 2, 3, 100);
  for (int i=0;i<rootPivots.size();i++)
  {
    rootPivots[i] += joints[3] - joints[0];
  }
  //subBlobs = kMeans(model.vertices, model.weights, rootPivots, 3, 3, 2);
  subBlobs = kMeans(model.vertices, model.weights, rootPivots, 3, 3, 1, true);
  for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma *= 0.6f; }
  for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma /= dsFac; }
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  rootPivots.resize(2);

  // nadpupek
  //subBlobs = kMeans(model.vertices, model.weights, pivots, 2, 6, 100);
  //subBlobs = kMeans(model.vertices, model.weights, std::vector<V3f>(), 4, 6, 100);
  for (int i=0;i<rootPivots.size();i++)
  {
    rootPivots[i] += joints[6] - joints[3];
  }
  //subBlobs = kMeans(model.vertices, model.weights, rootPivots, 3, 6, 2);
  subBlobs = kMeans(model.vertices, model.weights, rootPivots, rootPivots.size(), 6, 1, true);
  for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma *= 0.8f; } //0.6f; }
  for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma /= dsFac; }
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // hrudnik
  for (int i=0;i<rootPivots.size();i++)
  {
    rootPivots[i] += joints[9] - joints[6];
    rootPivots[i] += 0.4f*(joints[12] - joints[9]);
  }
  subBlobs = kMeans(model.vertices, model.weights, rootPivots, rootPivots.size(), 9, 1, true);
  for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma *= 0.6f; }
  // subBlobs = kMeans(model.vertices, model.weights, pivots, 2, 9, 100);
  // for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma /= dsFac; }
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // nadhrudi
  for (int i=0;i<rootPivots.size();i++)
  {
    rootPivots[i] += 0.6f*(joints[12] - joints[9]);
  }
  subBlobs = kMeans(model.vertices, model.weights, rootPivots, rootPivots.size(), 9, 1, true);
  for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma *= 0.8f; }//0.6f; }
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // leva spicka
  subBlobs = kMeans(model.vertices, model.weights, std::vector<V3f>(), 1, 10, 2);
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // prava spicka
  subBlobs = kMeans(model.vertices, model.weights, std::vector<V3f>(), 1, 11, 2);
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // leva lopatka
  pivots.clear();
  subBlobs = kMeans(model.vertices, model.weights, pivots, 1, 13, 2);
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // prava lopatka
  subBlobs = kMeans(model.vertices, model.weights, pivots, 1, 14, 2);
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // krk
  subBlobs = kMeans(model.vertices, model.weights, pivots, 1, 12, 2);
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // hlava
  subBlobs = kMeans(model.vertices, model.weights, pivots, 1, 15, 2);
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());
  {
    V3f headNeck = joints[15] - joints[12];
    V3f shoulderShoulder = joints[16] - joints[17];
    V3f m = headNeck / norm(headNeck);
    V3f n = shoulderShoulder / norm(shoulderShoulder);
    V3f o = cross(n,m);
    subBlobs[0].mu += 0.5f*subBlobs[0].sigma*o;
    subBlobs[0].sigma *= 0.7f;
    blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());
  }
  
  //subBlobs.resize(4);
  subBlobs.resize(3);

  // leva kycel
  placeBlobsOnBone(model, joints, 4, subBlobs);
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // prava kycel
  placeBlobsOnBone(model, joints, 5, subBlobs);
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  subBlobs.resize(5);
  // leve koleno
  placeBlobsOnBone(model, joints, 7, subBlobs);
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // prave koleno
  placeBlobsOnBone(model, joints, 8, subBlobs);
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  subBlobs.resize(4);
  // leve rameno
  placeBlobsOnBone(model, joints, 18, subBlobs);
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // prave rameno
  placeBlobsOnBone(model, joints, 19, subBlobs);
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // levy loket
  placeBlobsOnBone(model, joints, 20, subBlobs);
  for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma *= 1.5f; }
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // pravy loket
  placeBlobsOnBone(model, joints, 21, subBlobs);
  for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma *= 1.5f; }
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  subBlobs.resize(2);
  // levy kotnik
  placeBlobsOnBone(model, joints, 10, subBlobs);
  for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma *= 0.8f; }
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // pravy kotnik
  placeBlobsOnBone(model, joints, 11, subBlobs);
  for (int i=0;i<subBlobs.size();i++) { subBlobs[i].sigma *= 0.8f; }
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // leve zapesti
  subBlobs = kMeans(model.vertices, model.weights, pivots, 1, 20, 2);
  subBlobs[0].sigma *= 1.5f;
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  // prave zapesti
  subBlobs = kMeans(model.vertices, model.weights, pivots, 1, 21, 2);
  subBlobs[0].sigma *= 1.5f;
  blobs.insert(blobs.end(), subBlobs.begin(), subBlobs.end());

  blobWeights = A2f(model.joints.size(), blobs.size());
  for (int i=0;i<blobWeights.numel();i++)
  {
    blobWeights[i] = 0.0f;
  }

  colors = std::vector<V3f>(blobs.size(), V3f(1.0f,0.0f,0.0f));
  hsvColors = std::vector<V3f>(blobs.size(), V3f(0.0f, 0.0f, 0.0f));

  updateWeights(model.vertices, model.weights);
}

//============== END OF BLOB MODEL ==================

V2f rotate(const V2f& x,float angle)
{
  return V2f(x(0)*cos(angle)-x(1)*sin(angle),
             x(0)*sin(angle)+x(1)*cos(angle));  
}

V2f canvas2viewport(const View2D& view,const V2f& x)
{
  return rotate(x-V2f(view.panX,view.panY),view.angle)*view.zoom+0.5f*V2f(view.width,view.height);
}

V2f viewport2canvas(const View2D& view,const V2f& x)
{
  return rotate((x-0.5f*V2f(view.width,view.height))/view.zoom,-view.angle)+V2f(view.panX,view.panY);
}

View2D zoomAt(const View2D& view,const V2f& x,float zoom)
{
  const V2f pan = -(rotate(canvas2viewport(view,x),-view.angle) - zoom*x - rotate(0.5f*V2f(view.width,view.height),-view.angle))/zoom;
  
  View2D newView = view;
  newView.zoom = zoom;
  newView.panX = pan(0);
  newView.panY = pan(1);
  
  return newView;
}

View2D viewFitCanvas(const View2D& view,int canvasWidth,int canvasHeight,int padding=0)
{
  const int w = canvasWidth;
  const int h = canvasHeight;

  const V2f corners[4] = { V2f(0,0),
                           V2f(w,0),
                           V2f(0,h),
                           V2f(w,h) };

  float xmin = +FLT_MAX;
  float ymin = +FLT_MAX;

  float xmax = -FLT_MAX;
  float ymax = -FLT_MAX;

  for (int i=0;i<4;i++)
  {
    const V2f xy = rotate(corners[i],view.angle);
    xmin = std::min(xmin,xy(0));
    ymin = std::min(ymin,xy(1));
    xmax = std::max(xmax,xy(0));
    ymax = std::max(ymax,xy(1));
  }
  
  const float rotatedWidth  = xmax-xmin;
  const float rotatedHeight = ymax-ymin;
  
  View2D newView = view;
  
  newView.panX = float(canvasWidth)/2.0f;
  newView.panY = float(canvasHeight)/2.0f;
  
  newView.zoom = std::min(float(view.width-padding)/float(rotatedWidth),
                     float(view.height-padding)/float(rotatedHeight));

  return newView;
}

void rq3(const Eigen::Matrix3d &A, Eigen::Matrix3d &R, Eigen::Matrix3d& Q)
{
  // Find rotation Qx to set A(2,1) to 0
  double c = -A(2,2)/sqrt(A(2,2)*A(2,2)+A(2,1)*A(2,1));
  double s = A(2,1)/sqrt(A(2,2)*A(2,2)+A(2,1)*A(2,1));
  Eigen::Matrix3d Qx,Qy,Qz;
  Qx << 1 ,0, 0,0,c,-s, 0,s,c;
  R = A*Qx;
  // Find rotation Qy to set A(2,0) to 0
  c = R(2,2)/sqrt(R(2,2)*R(2,2)+R(2,0)*R(2,0) );
  s = R(2,0)/sqrt(R(2,2)*R(2,2)+R(2,0)*R(2,0) );
  Qy << c, 0, s, 0, 1, 0,-s, 0, c;
  R*=Qy;

  // Find rotation Qz to set A(1,0) to 0
  c = -R(1,1)/sqrt(R(1,1)*R(1,1)+R(1,0)*R(1,0));
  s =  R(1,0)/sqrt(R(1,1)*R(1,1)+R(1,0)*R(1,0));
  Qz << c ,-s, 0, s ,c ,0, 0, 0 ,1;
  R*=Qz;

  Q = Qz.transpose()*Qy.transpose()*Qx.transpose();
  // Adjust R and Q so that the diagonal elements of R are +ve
  for (int n=0; n<3; n++)
  {
    if (R(n,n)<0)
    {
      R.col(n) = - R.col(n);
      Q.row(n) = - Q.row(n);
    }
  }
}

void decomposePMatrix(const Eigen::Matrix<double,3,4> &P,
                      Eigen::Matrix3d& R,
                      Eigen::Matrix3d& K,
                      Eigen::Vector3d& C,
                      Eigen::Vector3d& t)
{
  Eigen::Matrix3d M = P.topLeftCorner<3,3>();
  Eigen::Vector3d m3 = M.row(2).transpose();
  // Follow the HartleyZisserman - "Multiple view geometry in computer vision" implementation chapter 3

  Eigen::Matrix3d P123,P023,P013,P012;
  P123 << P.col(1),P.col(2),P.col(3);
  P023 << P.col(0),P.col(2),P.col(3);
  P013 << P.col(0),P.col(1),P.col(3);
  P012 << P.col(0),P.col(1),P.col(2);

  double X = P123.determinant();
  double Y = -P023.determinant();
  double Z = P013.determinant();
  double T = -P012.determinant();
  C << X/T,Y/T,Z/T;

  R = K = Eigen::Matrix3d::Identity();
  rq3(M,K,R);
  K/=K(2,2); // EXTREMELY IMPORTANT TO MAKE THE K(2,2)==1 !!!
  // http://ksimek.github.io/2012/08/14/decompose/
  // Negate the second column of K and R because the y window coordinates and camera y direction are opposite is positive
  // This is the solution I've found personally to correct the behaviour using OpenGL gluPerspective convention
  //R.row(2)=-R.row(2);
  R.row(2)=-R.row(2);

  // t is the location of the world origin in camera coordinates.
  t = -R*C;    
}

void decomposePMatrix2(const Eigen::Matrix<double,3,4> &P,
                      Eigen::Matrix3d& R,
                      Eigen::Matrix3d& K,
                      Eigen::Vector3d& C,
                      Eigen::Vector3d& t)
{
  Eigen::Matrix3d M = P.topLeftCorner<3,3>();
  Eigen::Vector3d m3 = M.row(2).transpose();
  // Follow the HartleyZisserman - "Multiple view geometry in computer vision" implementation chapter 3

  Eigen::Matrix3d P123,P023,P013,P012;
  P123 << P.col(1),P.col(2),P.col(3);
  P023 << P.col(0),P.col(2),P.col(3);
  P013 << P.col(0),P.col(1),P.col(3);
  P012 << P.col(0),P.col(1),P.col(2);

  double X = P123.determinant();
  double Y = -P023.determinant();
  double Z = P013.determinant();
  double T = -P012.determinant();
  C << X/T,Y/T,Z/T;

  R = K = Eigen::Matrix3d::Identity();
  rq3(M,K,R);
  K/=K(2,2); // EXTREMELY IMPORTANT TO MAKE THE K(2,2)==1 !!!
  // http://ksimek.github.io/2012/08/14/decompose/
  // Negate the second column of K and R because the y window coordinates and camera y direction are opposite is positive
  // This is the solution I've found personally to correct the behaviour using OpenGL gluPerspective convention
  //R.row(2)=-R.row(2);

  // t is the location of the world origin in camera coordinates.
  t = -R*C;    
}

void getMatrices(const Mat3x4f& Pjzq,Mat4x4f& proj,Mat4x4f& view)
{
  Eigen::Matrix<double,3,4> P;

  for (int i=0;i<3;i++)
  for (int j=0;j<4;j++)
  {
    P(i,j) = Pjzq(i,j);
  }

  Eigen::Matrix3d R;
  Eigen::Matrix3d K;
  Eigen::Vector3d C;
  Eigen::Vector3d t;

  decomposePMatrix(P,R,K,C,t);

  Eigen::Affine3d OpenGLModelViewMatrix;
  OpenGLModelViewMatrix.setIdentity();
  OpenGLModelViewMatrix.linear().matrix() << R;
  OpenGLModelViewMatrix.translation() << t;

  for (int i=0;i<4;i++)
  for (int j=0;j<4;j++)
  {
    view(i,j) = OpenGLModelViewMatrix.matrix()(i,j);
  }   

  double znear = 0.1;
  double zfar = 1000;

  proj = Mat4x4f(K(0,0),K(0,1),-K(0,2), 0,
                     0,K(1,1),-K(1,2), 0,
                     0,0,znear+zfar,znear*zfar,
                     0,0,-1,0);
}

void glVertex(const V2f& v)
{
  glVertex2f(v(0),v(1));
}

void glVertex(const V3f& v)
{
  glVertex3f(v(0),v(1),v(2));
}

void glColor(const V3f& c)
{
  glColor3f(c(0),c(1),c(2));
}

float shade(const V3f& n,const V3f& lightDir)
{
  return (0.6f*pow(std::max(dot(normalize(n),normalize(lightDir)),0.0f),1.5f)+0.4f);
}

void normalize(GLfloat* v)
{
  float n = 1.0f/sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  v[0] *= n;
  v[1] *= n;
  v[2] *= n;
}

void drawSphereSubdiv(float* a,float* b,float* c,int div,float r,const V3f& center,const V3f& color,const V3f& lightDir)
{
  if (div<=0)
  {
    glColor(color*shade(V3f(a[0],a[1],a[2]),lightDir)); glVertex(r*V3f(a[0],a[1],a[2])+center);
    glColor(color*shade(V3f(b[0],b[1],b[2]),lightDir)); glVertex(r*V3f(b[0],b[1],b[2])+center);
    glColor(color*shade(V3f(c[0],c[1],c[2]),lightDir)); glVertex(r*V3f(c[0],c[1],c[2])+center);
  }
  else
  {
    float ab[3],ac[3],bc[3];
    for (int i=0;i<3;i++)
    {
      ab[i]=(a[i]+b[i])/2;
      ac[i]=(a[i]+c[i])/2;
      bc[i]=(b[i]+c[i])/2;
    }
    normalize(ab); normalize(ac); normalize(bc);
    drawSphereSubdiv(a,ab,ac,div-1,r,center,color,lightDir);
    drawSphereSubdiv(b,bc,ab,div-1,r,center,color,lightDir);
    drawSphereSubdiv(c,ac,bc,div-1,r,center,color,lightDir);
    drawSphereSubdiv(ab,bc,ac,div-1,r,center,color,lightDir);
  }
}

void drawSphere(const V3f& center,float radius,const V3f& color,const V3f& lightDir)
{
  static float X =.525731112119133606;
  static float Z =.850650808352039932;
  static float vdata[12][3] = {
    {-X, 0.0, Z}, {X, 0.0, Z}, {-X, 0.0, -Z}, {X, 0.0, -Z},
    {0.0, Z, X}, {0.0, Z, -X}, {0.0, -Z, X}, {0.0, -Z, -X},
    {Z, X, 0.0}, {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0}
  };
  static GLuint tindices[20][3] = {
    {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},
    {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},
    {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6},
    {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11}
  };

  glBegin(GL_TRIANGLES);
  for (int i=0;i<20;i++) { drawSphereSubdiv(vdata[tindices[i][0]],vdata[tindices[i][1]],vdata[tindices[i][2]],2,radius,center,color,lightDir); }
  glEnd();
}

void drawCamera(const Camera& camera,float width,float height,float scale)
{
  const float s = 0.02f*scale;

  const Mat3x3f Rc = transpose(camera.R);
  const Vec3f x = Rc*Vec3f(1,0,0);
  const Vec3f y = Rc*Vec3f(0,1,0);
  const Vec3f z = Rc*Vec3f(0,0,1);

  glLineWidth(1);
  glBegin(GL_LINES);
    glColor3f(1,0,0);
    glVertex(camera.C);
    glVertex(camera.C+s*x);

    glColor3f(0,1,0);
    glVertex(camera.C);
    glVertex(camera.C+s*y);

    glColor3f(0,0,1);
    glVertex(camera.C);
    glVertex(camera.C+s*z);

    V3f x00 = camera.C+2.0f*s*unproject(camera,V2f(    0,     0));
    V3f x10 = camera.C+2.0f*s*unproject(camera,V2f(width,     0));
    V3f x01 = camera.C+2.0f*s*unproject(camera,V2f(    0,height));
    V3f x11 = camera.C+2.0f*s*unproject(camera,V2f(width,height));

    glColor3f(0.5,0.5,0.5);
    glVertex(camera.C);
    glVertex(x00);

    glVertex(camera.C);
    glVertex(x10);

    glVertex(camera.C);
    glVertex(x01);

    glVertex(camera.C);
    glVertex(x11);

    glVertex(x00);
    glVertex(x10);

    glVertex(x10);
    glVertex(x11);

    glVertex(x11);
    glVertex(x01);

    glVertex(x01);
    glVertex(x00);

  glEnd();
}

Vec3f glUnproject(const V2f& x)
{
  GLint viewport[4];
  GLdouble modelview[16];
  GLdouble projection[16];
  GLfloat winX, winY, winZ;
  GLdouble posX, posY, posZ;

  glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
  glGetDoublev( GL_PROJECTION_MATRIX, projection );
  glGetIntegerv( GL_VIEWPORT, viewport );

  winX = (float)x(0);
  winY = (float)viewport[3] - (float)x(1);
  winZ = 1;
  gluUnProject(winX,winY,winZ,modelview,projection,viewport,&posX,&posY,&posZ);

  return V3f(posX, posY, posZ);
}

float distanceToTriangleSqr(const Vec3f& x,const Vec3f& v0,const Vec3f& _v1,const Vec3f& _v2)
{
  Vec3f diff = v0 - x;
  Vec3f _edge0 = _v1 - v0;
  Vec3f _edge1 = _v2 - v0;
  float a00 = dot(_edge0,_edge0);
  float a01 = dot(_edge0,_edge1);
  float a11 = dot(_edge1,_edge1);
  float b0  = dot(diff,_edge0);
  float b1  = dot(diff,_edge1);
  float c   = dot(diff,diff);
  float det = abs(a00*a11 - a01*a01);
  float s   = a01*b1 - a11*b0;
  float t   = a01*b0 - a00*b1;
  float sqrDistance;

  //if (a00<10e-7 || a11<10e-7 || det<10e-7) { return FLT_MAX; }

  if (s + t <= det)
  {
    if (s < 0.0)
    {
      if (t < 0.0)  // region 4
      {
        if (b0 < 0.0)
        {
          t = 0.0;
          if (-b0 >= a00)
          {
            s = 1.0;
            sqrDistance = a00 + (2.0)*b0 + c;
          }
          else
          {
            s = -b0/a00;
            sqrDistance = b0*s + c;
          }
        }
        else
        {
          s = 0.0;
          if (b1 >= 0.0)
          {
            t = 0.0;
            sqrDistance = c;
          }
          else if (-b1 >= a11)
          {
            t = 1.0;
            sqrDistance = a11 + (2.0)*b1 + c;
          }
          else
          {
            t = -b1/a11;
            sqrDistance = b1*t + c;
          }
        }
      }
      else  // region 3
      {
        s = 0.0;
        if (b1 >= 0.0)
        {
          t = 0.0;
          sqrDistance = c;
        }
        else if (-b1 >= a11)
        {
          t = 1.0;
          sqrDistance = a11 + (2.0)*b1 + c;
        }
        else
        {
          t = -b1/a11;
          sqrDistance = b1*t + c;
        }
      }
    }
    else if (t < 0.0)  // region 5
    {
      t = 0.0;
      if (b0 >= 0.0)
      {
        s = 0.0;
        sqrDistance = c;
      }
      else if (-b0 >= a00)
      {
        s = 1.0;
        sqrDistance = a00 + (2.0)*b0 + c;
      }
      else
      {
        s = -b0/a00;
        sqrDistance = b0*s + c;
      }
    }
    else  // region 0
    {
      // minimum at interior point
      float invDet = (1.0)/det;

      s *= invDet;
      t *= invDet;
      sqrDistance = s*(a00*s + a01*t + (2.0)*b0) +
                    t*(a01*s + a11*t + (2.0)*b1) + c;
    }
  }
  else
  {
    float tmp0, tmp1, numer, denom;

    if (s < 0.0)  // region 2
    {
      tmp0 = a01 + b0;
      tmp1 = a11 + b1;
      if (tmp1 > tmp0)
      {
        numer = tmp1 - tmp0;
        denom = a00 - (2.0)*a01 + a11;
        if (numer >= denom)
        {
          s = 1.0;
          t = 0.0;
          sqrDistance = a00 + (2.0)*b0 + c;
        }
        else
        {
          s = numer/denom;
          t = 1.0 - s;
          sqrDistance = s*(a00*s + a01*t + (2.0)*b0) +
                        t*(a01*s + a11*t + (2.0)*b1) + c;
        }
      }
      else
      {
        s = 0.0;
        if (tmp1 <= 0.0)
        {
          t = 1.0;
          sqrDistance = a11 + (2.0)*b1 + c;
        }
        else if (b1 >= 0.0)
        {
          t = 0.0;
          sqrDistance = c;
        }
        else
        {
          t = -b1/a11;
          sqrDistance = b1*t + c;
        }
      }
    }
    else if (t < 0.0)  // region 6
    {
      tmp0 = a01 + b1;
      tmp1 = a00 + b0;
      if (tmp1 > tmp0)
      {
        numer = tmp1 - tmp0;
        denom = a00 - (2.0)*a01 + a11;
        if (numer >= denom)
        {
          t = 1.0;
          s = 0.0;
          sqrDistance = a11 + (2.0)*b1 + c;
        }
        else
        {
          t = numer/denom;
          s = 1.0 - t;
          sqrDistance = s*(a00*s + a01*t + (2.0)*b0) +
                        t*(a01*s + a11*t + (2.0)*b1) + c;
        }
      }
      else
      {
        t = 0.0;
        if (tmp1 <= 0.0)
        {
          s = 1.0;
          sqrDistance = a00 + (2.0)*b0 + c;
        }
        else if (b0 >= 0.0)
        {
          s = 0.0;
          sqrDistance = c;
        }
        else
        {
          s = -b0/a00;
          sqrDistance = b0*s + c;
        }
      }
    }
    else  // region 1
    {
      numer = a11 + b1 - a01 - b0;
      if (numer <= 0.0)
      {
          s = 0.0;
          t = 1.0;
          sqrDistance = a11 + (2.0)*b1 + c;
      }
      else
      {
        denom = a00 - (2.0)*a01 + a11;
        if (numer >= denom)
        {
          s = 1.0;
          t = 0.0;
          sqrDistance = a00 + (2.0)*b0 + c;
        }
        else
        {
          s = numer/denom;
          t = 1.0 - s;
          sqrDistance = s*(a00*s + a01*t + (2.0)*b0) +
                        t*(a01*s + a11*t + (2.0)*b1) + c;
        }
      }
    }
  }

  // Account for numerical round-off error.
  if (sqrDistance < 0.0)
  {
    return 0.0;
  }

  return sqrDistance;
}

float distanceToBBoxSqr(const V3f& p,const nanort::BVHNode &node)
{
  float sqrDist = 0.0f;
  for (int i=0;i<3;i++)
  {
    float v = p[i];
    if (v<node.bmin[i]) { sqrDist += (node.bmin[i]-v)*(node.bmin[i]-v); }
    if (v>node.bmax[i]) { sqrDist += (v-node.bmax[i])*(v-node.bmax[i]); }
  }
  return sqrDist;
}

inline int getClosestTriangleID(const nanort::BVHAccel& nanoBVH,
                                const std::vector<V3f>& vertices,
                                const std::vector<V3i>& triangles,
                                const V3f& p)
{
  const float* vertices_data = (float*)vertices.data();
  const unsigned int* faces = (unsigned int*)triangles.data();
  
  int nodeStackIndex = 0;
  int nodeStack[512];
  nodeStack[0] = 0;

  float minDist = std::numeric_limits<float>::max();
  int minId = -1;

  const std::vector<nanort::BVHNode>& nodes = nanoBVH.GetNodes();
  const std::vector<unsigned int>& indices = nanoBVH.GetIndices();

  while (nodeStackIndex >= 0)
  {
    int index = nodeStack[nodeStackIndex];
    const nanort::BVHNode &node = nodes[index];

    nodeStackIndex--;

    if (node.flag == 0) // branch node
    {
      unsigned int childIds[2] = { node.data[0], node.data[1] };
      float childDists[2];

      for (int i=0;i<2;i++) { childDists[i] = distanceToBBoxSqr(p,nodes[childIds[i]]); }

      if (childDists[1]>childDists[0])
      {
        std::swap(childIds[0],childIds[1]);
        std::swap(childDists[0],childDists[1]);
      }

      for (int i=0;i<2;i++)
      {
        if (childDists[i]<=minDist) { nodeStack[++nodeStackIndex] = childIds[i]; }
      }
    }
    else // leaf node
    {
      unsigned int numTriangles = node.data[0];
      unsigned int offset = node.data[1];

      for (unsigned int i = 0; i < numTriangles; i++)
      {
        int id = indices[i + offset];

        const V3i& triangle = triangles[id];

        const V3f& v0 = vertices[triangle[0]];
        const V3f& v1 = vertices[triangle[1]];
        const V3f& v2 = vertices[triangle[2]];

        float dist = distanceToTriangleSqr(p,v0,v1,v2);
        if (dist < minDist)
        {
          minDist = dist;
          minId = id;
        }
      }
    }
  }
  assert(nodeStackIndex < 512);

  return minId;
}

V3f barycentric(const V3f& p,const V3f& a,const V3f& b,const V3f& c)
{
  const V3f v0 = b - a;
  const V3f v1 = c - a;
  const V3f v2 = p - a;
  const float d00 = dot(v0, v0);
  const float d01 = dot(v0, v1);
  const float d11 = dot(v1, v1);
  const float d20 = dot(v2, v0);
  const float d21 = dot(v2, v1);
  const float denom = d00 * d11 - d01 * d01;
  const float v = (d11 * d20 - d01 * d21) / denom;
  const float w = (d00 * d21 - d01 * d20) / denom;
  const float u = 1.0f - v - w;
  return V3f(u,v,w);
}

void genPoissonSamples(const std::vector<V3f>& vertices,
                       const std::vector<V3f>& normals,
                       const std::vector<V3i>& triangles,
                       const float radius,
                       std::vector<V3f>* out_points,
                       std::vector<V3i>* out_triangles,
                       std::vector<V3f>* out_barycentrics)
{
  std::vector<Utils_sampling::Vec3> verts(vertices.size());
  std::vector<Utils_sampling::Vec3> nors(vertices.size());
  std::vector<int> tris(triangles.size()*3);

  for (int i=0;i<vertices.size();i++)
  {
    const V3f v = vertices[i];
    const V3f n = normals[i];
    verts[i] = Utils_sampling::Vec3(v[0],v[1],v[2]);
    nors[i] = Utils_sampling::Vec3(n[0],n[1],n[2]);
  }

  for (int i=0;i<triangles.size();i++)
  {
    const V3i t = triangles[i];
    for (int j=0;j<3;j++) { tris[i*3+j] = t[j]; }
  }

  std::vector<Utils_sampling::Vec3> samplesPos;
  std::vector<Utils_sampling::Vec3> samplesNor;

  Utils_sampling::poisson_disk(radius,0,verts,nors,tris,samplesPos,samplesNor);

  std::vector<V3f>& points = *out_points;
  std::vector<V3i>& outTriangles = *out_triangles;
  std::vector<V3f>& outBarycentrics = *out_barycentrics;

  points = std::vector<V3f>(samplesPos.size());
  std::vector<V3f> outNormals = std::vector<V3f>(samplesNor.size());
  outTriangles = std::vector<V3i>(samplesPos.size());
  outBarycentrics = std::vector<V3f>(samplesPos.size());

  for (int i=0;i<points.size();i++)  { points[i] = V3f(samplesPos[i].x,samplesPos[i].y,samplesPos[i].z);  }
  for (int i=0;i<outNormals.size();i++) { outNormals[i] = V3f(samplesNor[i].x,samplesNor[i].y,samplesNor[i].z); }

  nanort::BVHBuildOptions nanoOptions;
  nanort::BVHAccel nanoBVH;
  nanoBVH.Build((float*)vertices.data(),(unsigned int*)triangles.data(),triangles.size(),nanoOptions);
  
  const int N = samplesPos.size();

  for (int i=0;i<N;i++)
  {
    const V3f& p = points[i];
    const V3f& n = outNormals[i];
    
    int triId = getClosestTriangleID(nanoBVH, vertices, triangles, p);
    if (triId > -1)
    {
      V3i tri = triangles[triId];
      V3f baryc = barycentric(p, vertices[tri(0)], vertices[tri(1)], vertices[tri(2)]);

      outTriangles[i] = tri;
      outBarycentrics[i] = baryc;
    }
    else
    {
      printf("BVH failed to find nearest triangle!\n");
      outTriangles[i] = V3i(0,0,0);
      outBarycentrics[i] = V3f(0,0,0);
    }
  }
}

Vec<3,ShortDual<3>> project(const Mat3x4f& PM, const V3f& x)
{
  V3f Px;
  for (int i=0;i<3;i++)
  {
    Px[i] = PM(i,0)*x(0) + PM(i,1)*x(1) + PM(i,2)*x(2) + PM(i,3);
  }

  Vec<3,ShortDual<3>> dPx;
  for (int i=0;i<3;i++)
  {
    dPx[i].a = Px[i];
    for (int j=0;j<3;j++)
    {
      dPx[i].b[j] = PM(i,j);
    }
  }

  return dPx;
}

float sqr(float x)
{
  return x*x;
}

template <int N, typename T>
void drawCircle(Array2<Vec<N,T> >& dst,int cx,int cy,int r,const Vec<N,T>& color)
{
  const int x0 = std::max(cx-r-1,0);
  const int x1 = std::min(cx+r+1,dst.width()-1);
  const int y0 = std::max(cy-r-1,0);
  const int y1 = std::min(cy+r+1,dst.height()-1);

  for (int y=y0;y<=y1;y++)
  for (int x=x0;x<=x1;x++)
  {
    if (x>=0 && y>=0 && x<dst.width() && y<dst.height())
    {
      float d = norm(V2f(x,y)-V2f(cx,cy));
      float a = 0.8f*(std::max(std::min(1.0f-abs(float(d)-float(r)),1.0f),0.0f));
      if (a>0.0f)
      {
        dst(x,y) = Vec<N,T>( ((1.0f-a)*Vec<N,float>(dst(x,y))) + ((a)*Vec<N,float>(color)) );
      }
    }
  }
}

template<typename T>
Vector<T> parametrizeTsAndAngles(const TRSA<T>& trsa)
{
  const TRS<T>& trs = trsa.trs;
  const std::vector<Vec<3,T>>& angles = trsa.angles;

  int cnt = 4;
  for (int i=0;i<angles.size();i++)
  {
    for (int j=0;j<3;j++)
    {
      if (jointMaxLimits[i][j] > jointMinLimits[i][j])
      {
        cnt++;
      }
    }
  }

  Vector<T> x(cnt);

  x[0] = trs.t(0);
  x[1] = trs.t(1);
  x[2] = trs.t(2);

  x[3] = trs.s;

  cnt = 4;
  for (int i=0;i<angles.size();i++)
  for (int j=0;j<3;j++)
  {
    if (jointMaxLimits[i][j] > jointMinLimits[i][j])
    {
      x[cnt++] = angles[i][j];
    }
  }
  
  return x;
}

template<typename T>
void deparametrizeTsAndAngles(const Vector<T>& x,TRSA<T>* out_trsa)
{
  std::vector<Vec<3,T>> angles(24);
  TRS<T>& trs = (*out_trsa).trs;

  trs.t(0) = x[0];
  trs.t(1) = x[1];
  trs.t(2) = x[2];
  
  trs.s    = x[3];

  int cnt = 4;
  for (int i=0;i<angles.size();i++)
  for (int j=0;j<3;j++)
  {
    angles[i][j] = (jointMaxLimits[i][j] > jointMinLimits[i][j]) ? x[cnt++] : T(jointMaxLimits[i][j]);
  }
  
  if (out_trsa) { (*out_trsa).angles = angles; }
}

template<typename T>
Vector<T> parametrizeTAndAngles(const TRSA<T>& trsa)
{
  int cnt = 3;
  for (int i=0;i<trsa.angles.size();i++)
  for (int j=0;j<3;j++)
  {
    if (jointMaxLimits[i][j] > jointMinLimits[i][j])
    {
      cnt++;
    }
  }
  
  Vector<T> x(cnt);

  x[0] = trsa.trs.t(0);
  x[1] = trsa.trs.t(1);
  x[2] = trsa.trs.t(2);

  cnt = 3;
  for (int i=0;i<trsa.angles.size();i++)
  for (int j=0;j<3;j++)
  {
    if (jointMaxLimits[i][j] > jointMinLimits[i][j])
    {
      x[cnt++] = trsa.angles[i][j];
    }
  }
  
  return x;
}

template<typename T>
void deparametrizeTAndAngles(const Vector<T>& x,TRSA<T>* out_trsa)
{
  std::vector<Vec<3,T>> angles(24);
  TRS<T>& trs = (*out_trsa).trs;

  trs.t(0) = x[0];
  trs.t(1) = x[1];
  trs.t(2) = x[2];
  
  int cnt = 3;
  for (int i=0;i<angles.size();i++)
  for (int j=0;j<3;j++)
  {
    angles[i][j] = (jointMaxLimits[i][j] > jointMinLimits[i][j]) ? x[cnt++] : T(jointMaxLimits[i][j]);
  }
  
  if (out_trsa) { (*out_trsa).angles = angles; }
}

struct TrackBlobsEnergy
{
  const std::vector<View>& views;
  const BlobModel& blobModel;
  const std::vector<Joint<float>>& joints;
  const TRS<float> trsStatic;
  std::vector<Vec<3,Dual<float>>> blobCenters;

  int iterCnt;

  TrackBlobsEnergy(const std::vector<View>& views,
                   const BlobModel& blobModel,
                   const std::vector<Joint<float>>& joints,
                   const TRS<float>& trsStatic)
  : views(views), blobModel(blobModel), joints(joints), trsStatic(trsStatic)
  {
    blobCenters.resize(blobModel.blobs.size());

    iterCnt = 0;
  }

  template<typename T,typename T2>
  void operator()(const Vector<T>& arg,T2& sum)
  {
    double t0;

    t0 = timerGet();
    TRSA<T> trsa;
    TRS<T>& trs = trsa.trs;
    std::vector<Vec<3,T>>& curAngles = trsa.angles;
    deparametrizeTAndAngles(arg,&trsa);
    trs.r(0) = trsStatic.r(0);
    trs.r(1) = trsStatic.r(1);
    trs.r(2) = trsStatic.r(2);
    trs.s = trsStatic.s;

    for (int i=1;i<curAngles.size();i++)
    for (int j=0;j<3;j++)
    {
      if (curAngles[i][j].a > jointMaxLimits[i][j])
      {
        sum += T(blobAngleLambda)*sqr(curAngles[i][j] - jointMaxLimits[i][j]);
      }
      if (curAngles[i][j].a < jointMinLimits[i][j])
      {
        sum += T(blobAngleLambda)*sqr(jointMinLimits[i][j] - curAngles[i][j]);
      }
    }
    
    // fix backbone
    {
      T backBoneLambda = T(0.1);
      Vec<3,T> avg = (curAngles[3] + curAngles[6] + curAngles[9]) / T(3.0f);
      Vec<3,T> diff;
      diff = curAngles[3] - avg;
      sum += backBoneLambda*dot(diff,diff);
      diff = curAngles[6] - avg;
      sum += backBoneLambda*dot(diff,diff);
      diff = curAngles[9] - avg;
      sum += backBoneLambda*dot(diff,diff);
    }
    // fix neck with head
    {
      T heckNeckLambda = T(0.1);
      Vec<3,T> avg = (curAngles[12] + curAngles[15]) / T(2.0f);
      Vec<3,T> diff;
      diff = curAngles[12] - avg;
      sum += heckNeckLambda*dot(diff,diff);
      diff = curAngles[15] - avg;
      sum += heckNeckLambda*dot(diff,diff);
    }

    const Mat<4,4,T> trsM = trsMatrix(trs);

    std::vector<V3f> orgAngles(curAngles.size());
    for (int i=0;i<orgAngles.size();i++) { orgAngles[i] = V3f(0,0,0); }

    std::vector<Mat<4,4,T>> Ms(joints.size());

    #pragma omp parallel for
    for (int j=0;j<joints.size();j++)
    {
      const Mat<4,4,float> M0 = evalJointTransform(j,joints,orgAngles);
      const Mat<4,4,T    > M1 = evalJointTransform(j,joints,curAngles);
      
      Ms[j] = trsM*(M1*Mat<4,4,T>(inverse(M0)));
    }

    std::vector<V3f> restBlobCenters(blobModel.blobs.size());
    #pragma omp parallel for
    for (int i=0;i<restBlobCenters.size();i++)
    {
      restBlobCenters[i] = blobModel.blobs[i].mu;
    }

    deformSamplePoints(restBlobCenters,Ms,blobModel.blobWeights,&blobCenters);

    #pragma omp parallel for
    for (int i=0;i<views.size();i++)
    {
      const View& view = views[i];

#ifdef DEBUG_BLOBS
      A2V3uc I = view.image;
#endif
      
      float f = (fabs(view.camera.K(0,0)) + fabs(view.camera.K(1,1))) / 2.0f;

      std::vector<Vec<2,ShortDual<3>>> mu2D(blobCenters.size());
      std::vector<ShortDual<3>> sigma2D(blobCenters.size());
      for (int j=0;j<blobCenters.size();j++)
      {
        Vec<3,ShortDual<3>> pdbc = project(view.camera.P, V3f(blobCenters[j][0].a, blobCenters[j][1].a, blobCenters[j][2].a));
        mu2D[j][0] = pdbc[0]/pdbc[2];
        mu2D[j][1] = pdbc[1]/pdbc[2];
        sigma2D[j] = blobModel.blobs[j].sigma * f / pdbc[2];
      }

      for (int k=0;k<view.imageBlobs.size();k++)
      {
        const ImageBlob& iBlob = view.imageBlobs[k];
        
        T2 iBlobSum;
        iBlobSum.a = 0.0f;
        iBlobSum.b = Vector<float>(sum.b.size());
        for (int i=0;i<iBlobSum.b.size();i++)
        {
          iBlobSum.b[i] = 0.0f;
        }
        
        for (int j=0;j<blobCenters.size();j++)
        {
          float dist2f = sqr(iBlob.mu[0] - mu2D[j][0].a) + sqr(iBlob.mu[1] - mu2D[j][1].a);
          if (dist2f > 4.0f*sqr(iBlob.sigma + sigma2D[j].a))
          {
            continue;
          }
          ShortDual<3> dist2 = sqr(iBlob.mu[0] - mu2D[j][0]) + sqr(iBlob.mu[1] - mu2D[j][1]);
          ShortDual<3> denom = sqr(iBlob.sigma) + sqr(sigma2D[j]);
          ShortDual<3> blobOverlap = 6.2832f*sqr(iBlob.sigma)*sqr(sigma2D[j])/denom * exp(-(dist2/denom));

          float colorSim = phi31(norm(iBlob.hsvColor - blobModel.hsvColors[j]));
          ShortDual<3> blobSim = colorSim*blobOverlap;

          iBlobSum.a += blobSim.a;
          for (int n=0;n<3;n++)
          {
            for (int m=0;m<blobCenters[j][n].b.n;m++)
            {
              iBlobSum.b[blobCenters[j][n].b.ind[m]] += blobSim.b[n] * blobCenters[j][n].b.val[m];
            }
          }
          
#ifdef DEBUG_BLOBS
          drawCircle(I,(int)xy[0].a,(int)xy[1].a,(int)sigma2D[j].a,V3uc(0,255,0));
#endif
        }
        float iBlobSelfOverlap = 2.0f*E(iBlob,iBlob);

        T2 sumPartial;
        sumPartial.a = 0.0f;
        sumPartial.b = Vector<float>(arg.size());
        for (int i=0;i<sumPartial.b.size();i++)
        {
          sumPartial.b[i] = 0.0f;
        }
        sumPartial -= std::min(iBlobSum,iBlobSelfOverlap) / view.sumBlobsOverlaps;

        #pragma omp critical
        {
          sum += sumPartial;
        }
      }

#ifdef DEBUG_BLOBS
      if (i==3)
      imwrite(I, spf("debug/blobs%d_%02d.png", i, iterCnt).c_str());
#endif
    }

    iterCnt++;
  }
};

struct AlignEnergy
{
  const std::vector<View>& views;
  const std::vector<Anchor>& anchors;
  const std::vector<V3f>& samplePoints;
  const A2f& sampleWeights;
  const std::vector<int>& fixedIndices;
  const std::vector<V3f>& fixedDeformedAnchorPoints;
  const std::vector<Joint<float>>& joints;
  const std::vector<Vec<3,float>>& prevAngles;
  const float lambdaAngle;
  const float scale;

  AlignEnergy(const std::vector<View>& views,
              const std::vector<Anchor>& anchors,
              const std::vector<V3f>& samplePoints,
              const A2f& sampleWeights,
              const std::vector<int>& fixedIndices,
              const std::vector<V3f>& fixedDeformedAnchorPoints,
              const std::vector<Joint<float>>& joints,
              const std::vector<Vec<3,float>>& prevAngles,
              const float lambdaAngle,
              const float scale=0.0f)
  : views(views),anchors(anchors),
    samplePoints(samplePoints),sampleWeights(sampleWeights),
    fixedIndices(fixedIndices),fixedDeformedAnchorPoints(fixedDeformedAnchorPoints),
    joints(joints),prevAngles(prevAngles),lambdaAngle(lambdaAngle),scale(scale)
  {
  }
    
  template<typename T,typename T2>
  void operator()(const Vector<T>& arg,T2& sum) const
  {
    TRSA<T> trsa;
    TRS<T>& trs = trsa.trs;
    std::vector<Vec<3,T>>& curAngles = trsa.angles;

    trs.r(0) = 0.0f;
    trs.r(1) = 0.0f;
    trs.r(2) = 0.0f;
    
    if (scale==0.0f)
    {
      deparametrizeTsAndAngles(arg,&trsa);
    }
    else
    {
      deparametrizeTAndAngles(arg,&trsa);
      trs.s = scale;
    }

    std::vector<V3f> orgAngles(curAngles.size());
    for (int i=0;i<orgAngles.size();i++) { orgAngles[i] = V3f(0,0,0); }

    std::vector<Mat<4,4,T>> Ms(joints.size());

    #pragma omp parallel for
    for (int j=0;j<joints.size();j++)
    {
      const Mat<4,4,float> M0 = evalJointTransform(j,joints,orgAngles);
      const Mat<4,4,T    > M1 = evalJointTransform(j,joints,curAngles);

      Ms[j] = M1*Mat<4,4,T>(inverse(M0));
    }
    
    if (fixedDeformedAnchorPoints.size()>0)
    {
      for (int i=0;i<curAngles.size();i++)
      {
        sum += T(1000)*sqr(curAngles[i][0]-prevAngles[i][0]);
        sum += T(1000)*sqr(curAngles[i][1]-prevAngles[i][1]);
        sum += T(1000)*sqr(curAngles[i][2]-prevAngles[i][2]);
        if (i>0) // do not regularize root joint rotation
        {
          sum += T(lambdaAngle)*(sqr(curAngles[i][0])+sqr(curAngles[i][1])+sqr(curAngles[i][2]));
        }

        for (int j=0;j<3;j++)
        {
          if (curAngles[i][j].a > jointMaxLimits[i][j])
          {
            sum += T(blobAngleLambda)*sqr(curAngles[i][j] - jointMaxLimits[i][j]);
          }
          if (curAngles[i][j].a < jointMinLimits[i][j])
          {
            sum += T(blobAngleLambda)*sqr(jointMinLimits[i][j] - curAngles[i][j]);
          }
        }
      }
    }
    
    Mat<4,4,T> trsM = trsMatrix(trs);
    
    // deform sample points
    #pragma omp parallel for
    for (int i=0;i<fixedDeformedAnchorPoints.size();i++)
    {
      const int idx = fixedIndices[i];
      const Vec<3,T> ov = Vec<3,T>(samplePoints[idx]);
      Vec<3,T> dv = Vec<3,T>(0,0,0);
      for (int j=0;j<joints.size();j++)
      {
        float w = sampleWeights(j, idx);
        if (w>0)
        {
          Vec<3,T> dvPart = p2e(Ms[j]*e2p(ov));
          dv += T(w)*dvPart;
        }
      }

      Vec<3,T> diff = Vec<3,T>(fixedDeformedAnchorPoints[i]) - p2e(trsM*e2p(dv));
      #pragma omp critical
      {
        sum += T(1000000.0)*dot(diff,diff);
      }
    }

    #pragma omp parallel for
    for (int i=0;i<anchors.size();i++)
    {
      const Anchor& anchor = anchors[i];
      
      const Vec<3,T> ov = Vec<3,T>(anchor.meshPoint);
      Vec<3,T> dv = Vec<3,T>(0,0,0);
      for (int j=0;j<joints.size();j++)
      {
        float w = anchor.jointWeights[j];
        if (w>0)
        {
          Vec<3,T> dvPart = p2e(Ms[j]*e2p(ov));        
          dv += T(w)*dvPart;
        }
      }
      
      Vec<2,T> diff = Vec<2,T>(anchor.viewPoint) - p2e(Mat<3,4,T>(views[anchor.viewId].camera.P)*trsM*e2p(Vec<3,T>(dv)));
      #pragma omp critical
      {
        sum += dot(diff,diff);
      }
    }
  }
};

template <typename F>
lbfgsfloatval_t lbfgsEvalGradAndValue(void* instance,
                                      const lbfgsfloatval_t* x,
                                      lbfgsfloatval_t* g,
                                      const int n,
                                      const lbfgsfloatval_t step)
{
  F& func = *((F*)instance);

  Vector<Dual<float>> arg(n);
  for (int i=0;i<n;i++)
  {
    arg(i).a = x[i];
    arg(i).b = SparseVector<float>(i);
  }

  DenseDual<float> value;
  value.a = 0;
  value.b = Vector<float>(n);
  for (int i=0;i<n;i++) { value.b[i] = 0; }

  func(arg,value);

  for (int i=0;i<n;i++)
  {
    g[i] = value.b[i];
  }

  return value.a;
}

template <typename F>
Vector<float> minimizeLBFGS(const F& f,const Vector<float>& x0,int maxIter=10000)
{
  const int n = x0.size();

  lbfgsfloatval_t fx;
  lbfgsfloatval_t* x = lbfgs_malloc(n);
  lbfgs_parameter_t param;

  for (int i=0;i<n;i++) { x[i] = x0(i); }

  lbfgs_parameter_init(&param);
  param.max_iterations = maxIter;
  param.m = 5;
  //param.m = 100;

  lbfgs(n,x,&fx,lbfgsEvalGradAndValue<F>,0,(void*)(&f),&param);

  Vector<float> argmin(n);
  for (int i=0;i<n;i++) { argmin(i) = x[i]; }

  lbfgs_free(x);
  return argmin;
}

void genSamples(const std::vector<V3f>& vertices,
                const std::vector<V3i>& triangles,
                const A2f& weights,
                const std::vector<V3f>& normals,
                const float radius,
                MeshPointSamples* pointSamples)
{
  genPoissonSamples(vertices,
                    normals,
                    triangles,
                    radius,
                    &(pointSamples->points),
                    &(pointSamples->triangles),
                    &(pointSamples->barycentrics));

  std::vector<V3f>& samplePoints = pointSamples->points;
  std::vector<V3i>& sampleTriangles = pointSamples->triangles;
  std::vector<V3f>& sampleBarycentrics = pointSamples->barycentrics;

  A2f& sampleWeights = pointSamples->weights;

  // Compute weights
  const int numWeightsPerVertex = weights.width();
  sampleWeights = A2f(numWeightsPerVertex, samplePoints.size());
  for (int i=0;i<sampleWeights.numel(); i++)
  {
    sampleWeights[i] = 0.0f;
  }
  for (int i=0;i<sampleWeights.height();i++)
  {
    const V3i& tri = sampleTriangles[i];
    const V3f& baryc = sampleBarycentrics[i];

    for (int j=0;j<3;j++)
    for (int k=0;k<numWeightsPerVertex;k++)
    {
      sampleWeights(k, i) += baryc[j]*weights(k, tri[j]);
    }
  }
}

void predicatePose(TRSAAnim* inout_trsaAnim, int frame, int predDir, float predVelocity)
{
  TRSAAnim& trsaAnim = *inout_trsaAnim;

  if (predDir > 0)
  {
    if ((frame > 0) && (frame < trsaAnim.size()))
    {
      TRS<float> trsPred = trsaAnim[frame-1].trs;
      std::vector<V3f> anglesPred = trsaAnim[frame-1].angles;

      if (frame > 1)
      {
        trsPred.t += predVelocity*(trsPred.t - trsaAnim[frame-2].trs.t);
        trsPred.r += predVelocity*(trsPred.r - trsaAnim[frame-2].trs.r);

        for (int i=0;i<anglesPred.size();i++)
        {
          anglesPred[i] += predVelocity*(anglesPred[i] - trsaAnim[frame-2].angles[i]);
        }
      }
      trsaAnim[frame].trs = trsPred;
      trsaAnim[frame].angles = anglesPred;
    }
  }
  if (predDir < 0)
  {
    if ((frame >= 0) && (frame < trsaAnim.size()-1))
    {
      TRS<float> trsPred = trsaAnim[frame+1].trs;
      std::vector<V3f> anglesPred = trsaAnim[frame+1].angles;

      if (frame < trsaAnim.size()-2)
      {
        trsPred.t += predVelocity*(trsPred.t - trsaAnim[frame+2].trs.t);
        trsPred.r += predVelocity*(trsPred.r - trsaAnim[frame+2].trs.r);

        for (int i=0;i<anglesPred.size();i++)
        {
          anglesPred[i] += predVelocity*(anglesPred[i] - trsaAnim[frame+2].angles[i]);
        }
      }
      trsaAnim[frame].trs = trsPred;
      trsaAnim[frame].angles = anglesPred;
    }
  }
}

static float LUT[256*256];

#define SQR(A) ((A)*(A))
#define DIST(A,B) ((SQR((int)(A[i](0))-(int)(B[i](0)))\
                   +SQR((int)(A[i](1))-(int)(B[i](1)))\
                   +SQR((int)(A[i](2))-(int)(B[i](2))))/3)

void initGetMaskLUT()
{
  for (int i=0;i<256*256;i++) LUT[i]=255*8*SQR(0.25*(float)exp(-0.0004*i));
}

A2uc getMask(const A2V3uc& image,const A2V3uc& background,const float threshold, const AABB& bbox)
{
  // Mask out the background according to paper Background Cut
  A2uc mask(size(image));
  A2i dat(size(image));

  int x0 = (int)bbox.bboxMin[0];
  int x1 = (int)bbox.bboxMax[0];
  int y0 = (int)bbox.bboxMin[1];
  int y1 = (int)bbox.bboxMax[1];

  #pragma omp parallel for
  for (int i=0;i<mask.numel();i++)
  {
    mask[i] = 0;
    dat[i]  = 255*255;
  }

  #pragma omp parallel for
  for (int y=y0;y<=y1;y++)
  for (int x=x0;x<=x1;x++)
  {
    int i = x+y*dat.width();
    dat[i] = DIST(image,background);
  }

  #pragma omp parallel for
  for (int y=y0;y<y1;y++)
  for (int x=x0;x<x1;x++)
  {
    const float vx=LUT[std::max(dat(x,y),dat(x+1,y))];
    const float vy=LUT[std::max(dat(x,y),dat(x,y+1))];

    mask(x,y) = (255-vx-vy>threshold) ? 255 : 0;
  }

  #pragma omp parallel for
  for (int x=0;x<image.width();x++) mask(x,image.height()-1) = 0;
  #pragma omp parallel for
  for (int y=0;y<image.height();y++) mask(image.width()-1,y) = 0;

  return mask;
}

float phi31(float x)
{
  return abs(x) < 1.0f ? (std::max(std::pow(1.0f-abs(x),4.0f),0.0f)*(1.0f+4.0f*abs(x))) : 0.0f;
}

float E(const ImageBlob& bi,const ImageBlob& bj)
{
  const float dist2 = sqr(bi.mu(0)-bj.mu(0))+
                      sqr(bi.mu(1)-bj.mu(1));

  if (dist2>sqr(3.0f*(bi.sigma+bj.sigma))) { return 0; }

  const float denom = (sqr(bi.sigma)+sqr(bj.sigma));

  return 6.2832f*((sqr(bi.sigma)*sqr(bj.sigma))/denom)*exp(-dist2/denom);
}

void genImageBlobs(const std::vector<AABB>& bboxes, int w, std::vector<View>* inout_views)
{
  std::vector<View>& views = *inout_views;

  #pragma omp parallel for
  for (int i=0;i<views.size();i++)
  {
    View& view = views[i];
    const A2V3uc& I = view.image;
    const A2uc& M = view.mask;
    view.imageBlobs.clear();

    int x0 = ((int)bboxes[i].bboxMin[0] / w) * w;
    int y0 = ((int)bboxes[i].bboxMin[1] / w) * w;
    int x1 = ((int)bboxes[i].bboxMax[0] / w) * w;
    int y1 = ((int)bboxes[i].bboxMax[1] / w) * w;  

    for (int y=y0;y<y1;y+=w)
    for (int x=x0;x<x1;x+=w)
    {
      ImageBlob blob;
      V3f c(0,0,0);
      int maskSum = 0;
      for (int yb=y;yb<y+w;yb++)
      for (int xb=x;xb<x+w;xb++)
      {
        c += V3f(I(xb,yb));
        maskSum += M(xb,yb);
      }

      if (maskSum)
      {
        blob.sigma = w/2.0f;
        blob.mu = V2f(x+blob.sigma,y+blob.sigma);
        blob.rgbColor = c / (w*w*255.0f);
        blob.hsvColor = hsvCone(rgb2hsv(blob.rgbColor));
        view.imageBlobs.push_back(blob);
      }
    }
  }

  #pragma omp parallel for
  for (int i=0;i<views.size();i++)
  {
    View& view = views[i];

    int W = view.image.width()  / w;
    int H = view.image.height() / w;
    A2uc mask(W,H);
    for (int j=0;j<mask.numel();j++)
    {
      mask[j] = 0;
    }

    for (int j=0;j<view.imageBlobs.size();j++)
    {
      const ImageBlob& iBlob = view.imageBlobs[j];
      mask((int)(iBlob.mu[0]-iBlob.sigma)/w,(int)(iBlob.mu[1]-iBlob.sigma)/w) = 1;
    }

    view.sumBlobsOverlaps = 0.0f;

    for (int y=0;y<H;y++)
    for (int x=0;x<W;x++)
    {
      if (mask(x,y))
      {
        ImageBlob bi;
        bi.sigma = w/2.0f;
        bi.mu = V2f(x*w,y*w);
        for (int dy=-3;dy<=3;dy++)
        for (int dx=-3;dx<=3;dx++)
        {
          int xj = x+dx;
          int yj = y+dy;
          if ((xj>=0)&&(xj<W)&&(yj>=0)&&(yj<H)&&mask(xj,yj))
          {
            ImageBlob bj;
            bj.sigma = w/2.0f;
            bj.mu = V2f(xj*w,yj*w);
            view.sumBlobsOverlaps += E(bi,bj);
          }
        }
      }
    }
  }
}

void loadViews(const std::vector<Video>& videos,
               const std::vector<Camera>& cameras,
               const int frame,
               const std::vector<A2V3uc>& bgs,
               const TRSAAnim& trsaAnim,
               const SkinningModel& model,
               const BlobModel& blobModel,
               float bgsThr,
               int imageBlobSize,
               std::vector<View>* out_views)
{
  std::vector<View>& views = *out_views;

  #pragma omp parallel for
  for (int i=0;i<views.size();i++)
  {
    const Video& video = videos[i];
    const Camera& camera = cameras[i];
    View& view = views[i];
    view.image = A2V3uc(vidGetWidth(video),vidGetHeight(video));
    vidGetFrameData(video,frame,(void*)view.image.data());
    view.camera = camera;
  }

  std::vector<AABB> bboxes = blobModel.getBBoxes(trsaAnim[frame], model, views);
  
  #pragma omp parallel for
  for (int i=0;i<views.size();i++)
  {
    View& view = views[i];
    view.mask = getMask(view.image, bgs[i], bgsThr, bboxes[i]);
  }

  genImageBlobs(bboxes, imageBlobSize, &views);
}

template<int M,int N>
bool readMat(FILE* f,Mat<M,N,float>* out_mat)
{
  Mat<M,N,float> mat;

  for (int i=0;i<M;i++)
  for (int j=0;j<N;j++)
  {
    if (fscanf(f,"%f",&mat(i,j))!=1) { return false; }
  }

  if (out_mat) { *out_mat = mat; }
  
  return true;
}

bool readCamera(const char* fileName,Camera* out_camera)
{
  Mat3x4f P;

  FILE* f = fopen(fileName,"r");
  if (!f) goto bail;

  if (!readMat(f,&P)) { goto bail; }
  if (out_camera) { *out_camera = Camera(P); }
  return true;

bail:
  if (f) { fclose(f); }
  return false;
}

void initJointLimits(const SkinningModel& model, std::vector<V3f>* out_jointMinLimits, std::vector<V3f>* out_jointMaxLimits)
{
  std::vector<V3f>& jointMinLimits = *out_jointMinLimits;
  std::vector<V3f>& jointMaxLimits = *out_jointMaxLimits;
  
  jointMinLimits = std::vector<V3f>(model.joints.size(), V3f(-M_PI, -M_PI, -M_PI));
  jointMaxLimits = std::vector<V3f>(model.joints.size(), V3f(+M_PI, +M_PI, +M_PI));

  jointMinLimits[4]  = V3f(-M_PI*2/3,0.0f,  0.0f); // left knee
  jointMinLimits[5]  = V3f(-M_PI*2/3,0.0f,  0.0f); // right knee
  //// jointMinLimits[7]  = V3f(-M_PI/4,  0.0f,  0.0f); // left ankle
  //// jointMinLimits[8]  = V3f(-M_PI/4,  0.0f,  0.0f); // right ankle
  jointMinLimits[7]  = V3f(0.0f,  0.0f,  0.0f); // left ankle
  jointMinLimits[8]  = V3f(0.0f,  0.0f,  0.0f); // right ankle
  jointMinLimits[10] = V3f( 0.0f,  0.0f,  0.0f); // left toes
  jointMinLimits[11] = V3f( 0.0f,  0.0f,  0.0f); // right toes
  // jointMinLimits[18] = V3f( 0.0f,  0.0f,  0.0f); // left elbow
  // jointMinLimits[19] = V3f( 0.0f,-M_PI*2/3,0.0f); // right elbow
  jointMinLimits[18] = V3f( 0.0f,-M_PI*0.9f,0.0f); // left elbow
  jointMinLimits[19] = V3f( 0.0f,-M_PI*0.9f,0.0f); // right elbow
  jointMinLimits[20] = V3f( 0.0f,  0.0f,  0.0f); // left wrist
  jointMinLimits[21] = V3f( 0.0f,  0.0f,  0.0f); // right wrist
  jointMinLimits[22] = V3f( 0.0f,  0.0f,  0.0f); // left fingers
  jointMinLimits[23] = V3f( 0.0f,  0.0f,  0.0f); // right fingers

  jointMaxLimits[4]  = V3f( 0.0f,  0.0f,  0.0f); // left knee
  jointMaxLimits[5]  = V3f( 0.0f,  0.0f,  0.0f); // right knee
  //// jointMaxLimits[7]  = V3f(+M_PI/4,0.0f,  0.0f); // left ankle
  //// jointMaxLimits[8]  = V3f(+M_PI/4,0.0f,  0.0f); // right ankle
  jointMaxLimits[7]  = V3f(0.0f ,0.0f,  0.0f); // left ankle
  jointMaxLimits[8]  = V3f(0.0f ,0.0f,  0.0f); // right ankle
  jointMaxLimits[10] = V3f( 0.0f,  0.0f,  0.0f); // left toes
  jointMaxLimits[11] = V3f( 0.0f,  0.0f,  0.0f); // right toes
  // jointMaxLimits[18] = V3f( 0.0f,+M_PI*2/3,0.0f); // left elbow
  // jointMaxLimits[19] = V3f( 0.0f,  0.0f,  0.0f); // right elbow
  jointMaxLimits[18] = V3f( 0.0f,+M_PI*0.9f,0.0f); // left elbow
  jointMaxLimits[19] = V3f( 0.0f,+M_PI*0.9f,0.0f); // right elbow
  jointMaxLimits[20] = V3f( 0.0f,  0.0f,  0.0f); // left wrist
  jointMaxLimits[21] = V3f( 0.0f,  0.0f,  0.0f); // right wrist
  jointMaxLimits[22] = V3f( 0.0f,  0.0f,  0.0f); // left fingers
  jointMaxLimits[23] = V3f( 0.0f,  0.0f,  0.0f); // right fingers

  // jointMinLimits[1]  = V3f(-M_PI/4,-M_PI/4,-M_PI/4); // left hip
  // jointMinLimits[2]  = V3f(-M_PI/4,-M_PI/4,   0.0f); // right hip

  // jointMaxLimits[1]  = V3f(+M_PI/2,+M_PI/4,   0.0f); // left hip
  // jointMaxLimits[2]  = V3f(+M_PI/2,+M_PI/4,+M_PI/4); // right hip

  jointMinLimits[1]  = V3f(-M_PI*0.7f,-M_PI*0.7f,-M_PI*0.7f); // left hip
  jointMinLimits[2]  = V3f(-M_PI*0.7f,-M_PI*0.7f,-M_PI*0.7f); // right hip

  jointMaxLimits[1]  = V3f(+M_PI*0.7f,+M_PI*0.7f,+M_PI*0.7f); // left hip
  jointMaxLimits[2]  = V3f(+M_PI*0.7f,+M_PI*0.7f,+M_PI*0.7f); // right hip

  jointMinLimits[3]  = V3f(-M_PI/4,-M_PI/4,-M_PI/8); // chest lower
  jointMaxLimits[3]  = V3f(+M_PI/8,+M_PI/4,+M_PI/8); // chest lower

  jointMinLimits[6]  = V3f(-M_PI/4,-M_PI/4,-M_PI/8); // chest middle
  jointMaxLimits[6]  = V3f(+M_PI/8,+M_PI/4,+M_PI/8); // chest middle

  jointMinLimits[9]  = V3f(-M_PI/4,-M_PI/4,-M_PI/8); // chest upper
  jointMaxLimits[9]  = V3f(+M_PI/8,+M_PI/4,+M_PI/8); // chest upper

  jointMinLimits[12] = V3f(-M_PI/8,-M_PI/8,-M_PI/8); // neck
  jointMaxLimits[12] = V3f(+M_PI/8,+M_PI/8,+M_PI/8); // neck

  // jointMinLimits[13] = V3f(-M_PI/8,   0.0f,-M_PI/8); // left clavicle
  // jointMinLimits[14] = V3f(-M_PI/8,-M_PI/4,-M_PI/8); // right clavicle

  // jointMaxLimits[13] = V3f(+M_PI/8,+M_PI/4,+M_PI/8); // left clavicle
  // jointMaxLimits[14] = V3f(+M_PI/8,   0.0f,+M_PI/8); // right clavicle
  jointMinLimits[13] = V3f(0.0f,0.0f,0.0f); // left clavicle
  jointMinLimits[14] = V3f(0.0f,0.0f,0.0f); // right clavicle

  jointMaxLimits[13] = V3f(0.0f,0.0f,0.0f); // left clavicle
  jointMaxLimits[14] = V3f(0.0f,0.0f,0.0f); // right clavicle

  jointMinLimits[15] = V3f(-M_PI/8,-M_PI/8,-M_PI/8); // head
  jointMaxLimits[15] = V3f(+M_PI/8,+M_PI/8,+M_PI/8); // head

  // jointMinLimits[16] = V3f(-M_PI*3/8,-M_PI/4,-M_PI*3/8); // left shoulder
  // jointMinLimits[17] = V3f(-M_PI*3/8,-M_PI/2,-M_PI/2); // right shoulder

  // jointMaxLimits[16] = V3f(+M_PI*3/8,+M_PI/2,+M_PI/2); // left shoulder
  // jointMaxLimits[17] = V3f(+M_PI*3/8,+M_PI/4,+M_PI*3/8); // right shoulder
  jointMinLimits[16] = V3f(-M_PI*0.6f,-M_PI*0.6f,-M_PI*0.6f); // left shoulder
  jointMinLimits[17] = V3f(-M_PI*0.6f,-M_PI*0.6f,-M_PI*0.6f); // right shoulder

  jointMaxLimits[16] = V3f(+M_PI*0.6f,+M_PI*0.6f,+M_PI*0.6f); // left shoulder
  jointMaxLimits[17] = V3f(+M_PI*0.6f,+M_PI*0.6f,+M_PI*0.6f); // right shoulder
}

void doViewNavig(const Mode mode, View2D* inout_view)
{
  View2D& view = *inout_view;

  static int lastMouseX;
  static int lastMouseY;

  static int dragStartX;
  static int dragStartY;
  static View2D dragStartView;

  static View2D lastView;
  
  if (mode==ModePan)
  {
    if (mouseDown(ButtonLeft))
    {
      lastMouseX = mouseX();
      lastMouseY = mouseY();

      dragStartX = mouseX();
      dragStartY = mouseY();
      dragStartView = view;
    }

    if (mousePressed(ButtonLeft))
    {
      const V2f delta = viewport2canvas(dragStartView,V2f(dragStartX,dragStartY)) - 
                        viewport2canvas(dragStartView,V2f(mouseX(),mouseY()));

      view.panX = dragStartView.panX + delta(0);
      view.panY = dragStartView.panY + delta(1);
    }    
  }
  if (mouseWheelDelta()!=0 /*&& keyPressed(KeyAlt)*/)
  {
    view = zoomAt(view,viewport2canvas(view,V2f(mouseX(),mouseY())),view.zoom*pow(1.1f,float(mouseWheelDelta())/120.0f));
  }
}

bool hideVideo = false;
bool showCameras = false;
bool showMesh = false;
bool showJoints = false;
bool showBlobModel = false;
bool showBGSMask = false;
bool showImageBlobs = false;

ModelViewMode modelViewMode = MODEL_VIEW_MODE_NORMALS;
Mode mode = ModeNone;

bool optimizeScale = false;
int optIters = 60;
float predVelocity = 0.3f;
float bgsThr = 96.0f;
int imageBlobSize = 4;

int track = 0;
bool fitModel = false;
bool playback = false;

float angleLambda = 1000.0f;
float blobAngleLambda = 1.0f;

std::vector<A2V3uc> bgs;

std::vector<V3f> jointMinLimits;
std::vector<V3f> jointMaxLimits;

std::vector<V3f> prevAngles;
std::vector<Anchor> anchors;

BBoxSelector bboxSelector;
std::vector<int> fixedIndices;
std::vector<V3f> fixedDeformedAnchorPoints;
MeshPointSamples anchorSamples;
std::vector<bool> vertexSelection;

TRSAAnim trsaAnim;
BlobModel blobModel;

void doView(int frame,int numCameras,const std::vector<View>& views,std::vector<View2D>* inout_view2ds,int* inout_selView,SkinningModel& model,const std::vector<Video>& videos)
{
  int& selView = *inout_selView;
  std::vector<V3f>& curAngles = trsaAnim[frame].angles;

  if (keyDown(KeyLeft))  { selView = selView-1; if (selView<0) { selView = numCameras-1; } }
  if (keyDown(KeyRight)) { selView = (selView+1)%numCameras; }

  if (keyDown(KeyS)) { showMesh = !showMesh; }
  if (keyDown(KeyJ)) { showJoints = !showJoints; }
  if (keyDown(KeyB)) { showBlobModel = !showBlobModel; }

  const View& view = views[selView];
  View2D& view2d = (*inout_view2ds)[selView];

  if (view2d.width!=widgetWidth() ||
      view2d.height!=widgetHeight())
  {
    view2d.width = widgetWidth();
    view2d.height = widgetHeight();
    view2d = viewFitCanvas(view2d,view.image.width(),view.image.height(),8);
  }
  
  if (keyDown(KeyHome))
  {
    view2d = viewFitCanvas(view2d,view.image.width(),view.image.height(),8);
  }

  const V2f mouseXY = viewport2canvas(view2d,V2f(mouseX(),mouseY()));
  
  float n = 0.1f;
  float f = 1000.0f;
   
  Mat4x4f Mproj;
  Mat4x4f Mview;
  getMatrices(view.camera.P,Mproj,Mview);

  glViewport(0,0,widgetWidth(),widgetHeight());
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  const V2f topLeft = viewport2canvas(view2d,V2f(0,0));
  const V2f bottomRight = viewport2canvas(view2d,V2f(view2d.width,view2d.height));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(topLeft(0),bottomRight(0),bottomRight(1),topLeft(1), n, f);
  glMultMatrixf(transpose(Mproj).data());
  
  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf(transpose(Mview).data());
  const Ray ray = Ray(view.camera.C,glUnproject(V2f(mouseX(),mouseY())));

  ///////////////////////////////////////////////////////////////////////////////

  std::vector<V3f> orgAngles(model.joints.size());
  for (int i=0;i<orgAngles.size();i++) { orgAngles[i] = V3f(0,0,0); }
  const Mat4x4f M = trsMatrix(trsaAnim[frame].trs);

  std::vector<V3f> deformedJoints;
  {
    A2f jointWeights(model.joints.size(),model.joints.size());
    for (int i=0;i<jointWeights.numel();i++)
    {
      jointWeights[i] = 0.0f;
    }
    for (int i=0;i<model.joints.size();i++)
    {
      jointWeights(i,i) = 1.0f;
    }
    deformedJoints = deformVertices(model.getRestPoseJointPositions(),
                                    jointWeights,
                                    model.joints,
                                    orgAngles,
                                    model.joints,
                                    curAngles);

    for (int i=0;i<deformedJoints.size();i++)
    {
      deformedJoints[i] = p2e(M*e2p(deformedJoints[i]));
    }
  }

  std::vector<V3f> deformedVertices = deformVertices(model.vertices,
                                                     model.weights,
                                                     model.joints,
                                                     orgAngles,
                                                     model.joints,
                                                     curAngles);
  #pragma omp parallel for
  for (int i=0;i<deformedVertices.size();i++)
  {
    deformedVertices[i] = p2e(M*e2p(deformedVertices[i]));
  }
  
  int hitId = -1;
  float hitT = 0;
  double t0 = timerGet();
  bool hit = showMesh && intersectMesh(ray,deformedVertices,model.triangles,&hitId,&hitT);
  
  const V3f deformedHitPoint = ray.o+hitT*ray.d;

  std::vector<float> jointWeights(model.joints.size());
  for (int j=0;j<jointWeights.size();j++) { jointWeights[j] = 0; }

  V3f originalHitPoint = V3f(0,0,0);
  if (hit)
  {
    const V3i& triangle = model.triangles[hitId];
    const V3f baryCoords = barycentric(deformedHitPoint,deformedVertices[triangle[0]],deformedVertices[triangle[1]],deformedVertices[triangle[2]]);
    originalHitPoint  = model.vertices[triangle[0]]*baryCoords(0) + model.vertices[triangle[1]]*baryCoords(1) + model.vertices[triangle[2]]*baryCoords(2);
    
    for (int i=0;i<3;i++)
    for (int j=0;j<jointWeights.size();j++)
    {
      jointWeights[j] += baryCoords[i]*model.weights(j,triangle[i]);
    }
  }
  if (mode == ModeNone)
  {
    if (mouseDown(ButtonLeft)&&keyPressed(KeyF))    { mode = ModeFix; }
    if (mouseDown(ButtonLeft)&&!keyPressed(KeyF))   { mode = ModePan; }
    if (mouseDown(ButtonLeft)&&hit) { mode = ModePick; }
    if (mouseDown(ButtonRight)&&(anchors.size()>0)) { anchors.pop_back(); }
  }
  else
  {
    if (mouseUp(ButtonLeft))
    {
      if (mode == ModeFix)
      {
        if (!keyPressed(KeyShift))
        {
          fixedIndices.clear();
          fixedDeformedAnchorPoints.clear();
          #pragma omp parallel for
          for (int i=0;i<vertexSelection.size();i++)
          {
            vertexSelection[i] = false;
          }
        }

        std::vector<V3f> orgAngles(curAngles.size(), V3f(0,0,0));
        std::vector<V3f> deformedAnchorPoints = deformVertices(anchorSamples.points,anchorSamples.weights,model.joints,orgAngles,model.joints,curAngles);

        #pragma omp parallel for
        for (int i=0;i<deformedAnchorPoints.size();i++)
        {
          deformedAnchorPoints[i] = p2e(M*e2p(deformedAnchorPoints[i]));
        }

        // compute intersection of bounding box selection and sample points
        int viewId = bboxSelector.viewId;
        double t0 = timerGet();
      
        V3f dtl = normalize(unproject(views[viewId].camera,V2f(bboxSelector.bboxMin[0], bboxSelector.bboxMin[1])));
        V3f dtr = normalize(unproject(views[viewId].camera,V2f(bboxSelector.bboxMax[0], bboxSelector.bboxMin[1])));
        V3f dbr = normalize(unproject(views[viewId].camera,V2f(bboxSelector.bboxMax[0], bboxSelector.bboxMax[1])));
        V3f dbl = normalize(unproject(views[viewId].camera,V2f(bboxSelector.bboxMin[0], bboxSelector.bboxMax[1])));

        V3f tl = views[viewId].camera.C+dtl;
        V3f tr = views[viewId].camera.C+dtr;
        V3f br = views[viewId].camera.C+dbr;
        V3f bl = views[viewId].camera.C+dbl;

        V3f nt = normalize(cross(dtl,dtr));
        V3f nr = normalize(cross(dtr,dbr));
        V3f nb = normalize(cross(dbr,dbl));
        V3f nl = normalize(cross(dbl,dtl));

        for (int i=0;i<deformedAnchorPoints.size();i++)
        {
          const V3f& v = deformedAnchorPoints[i];

          // do not test the vertex if it's already in the set
          if (std::find(fixedIndices.begin(), fixedIndices.end(), i) != fixedIndices.end())
          {
            continue;
          }

          if (dot(nt,v-tl)<0)
          {
            continue;
          }
          if (dot(nr,v-tr)<0)
          {
            continue;
          }
          if (dot(nb,v-br)<0)
          {
            continue;
          }
          if (dot(nl,v-bl)<0)
          {
            continue;
          }

          fixedIndices.push_back(i);
          fixedDeformedAnchorPoints.push_back(v);
        }

        for (int i=0;i<vertexSelection.size();i++)
        {
          const V3f& v = deformedVertices[i];

          if (dot(nt,v-tl)<0)
          {
            continue;
          }
          if (dot(nr,v-tr)<0)
          {
            continue;
          }
          if (dot(nb,v-br)<0)
          {
            continue;
          }
          if (dot(nl,v-bl)<0)
          {
            continue;
          }

          vertexSelection[i] = true;
        }

        printf("AlignEnergy(): %fs, ||fixedAnchorPoints|| = %d\n", (timerGet() - t0)*0.001, (int)fixedDeformedAnchorPoints.size());
      }

      mode = ModeNone;
    }
  }
  //if (mouseUp(ButtonLeft)) { mode = ModePan; }
  // if (!hit)
  // {
  //   if (mouseDown(ButtonLeft)&&(!keyPressed(KeyControl)))    { mode = ModePan; }
  //   if (mousePressed(ButtonLeft)&&(!keyPressed(KeyControl))) { mode = ModePan; }
  // }
  doViewNavig(mode, &view2d);
  static V3f dragStartPoint3D;
  static std::vector<Anchor> dragStartPointPairs;
  static std::vector<float> dragStartJointWeights;

  if (mode==ModePick)
  {
    if (mouseDown(ButtonLeft))
    {
      dragStartPoint3D = originalHitPoint;
      dragStartPointPairs = anchors;
      dragStartJointWeights = jointWeights;
    }

    anchors = dragStartPointPairs;
    Anchor pp;
    pp.meshPoint = dragStartPoint3D;
    pp.viewPoint = viewport2canvas(view2d,V2f(mouseX(),mouseY()));
    pp.jointWeights = dragStartJointWeights;
    pp.viewId = selView;
    anchors.push_back(pp);
  }
  
  static V2f bboxStartPoint2D;
  if (mode==ModeFix)
  {
    bboxSelector.viewId = selView;
    if (mouseDown(ButtonLeft))
    {
      bboxStartPoint2D = viewport2canvas(view2d,V2f(mouseX(),mouseY()));
      bboxSelector.bboxMin = bboxStartPoint2D;
      bboxSelector.bboxMax = bboxStartPoint2D;
    }
    if (mousePressed(ButtonLeft))
    {
      V2f bboxCurrentPoint2D = viewport2canvas(view2d,V2f(mouseX(),mouseY()));
      bboxSelector.bboxMin = std::min(bboxStartPoint2D, bboxCurrentPoint2D);
      bboxSelector.bboxMax = std::max(bboxStartPoint2D, bboxCurrentPoint2D);
    }
  }

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(topLeft(0),bottomRight(0),bottomRight(1),topLeft(1));             
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

#ifdef USE_OPENGL
  static GLTexture2D texImage;
  if (showImageBlobs)
  {
    A2V3uc mask(view.image.size());
    #pragma omp parallel for
    for (int i=0;i<mask.numel();i++)
    {
      mask[i] = V3uc(0,0,0);
    }
    for (int i=0;i<view.imageBlobs.size();i++)
    {
      const ImageBlob& imageBlob = view.imageBlobs[i];
      int x0 = imageBlob.mu[0] - imageBlob.sigma;
      int y0 = imageBlob.mu[1] - imageBlob.sigma;
      int x1 = imageBlob.mu[0] + imageBlob.sigma;
      int y1 = imageBlob.mu[1] + imageBlob.sigma;
      for (int y=y0;y<y1;y++)
      for (int x=x0;x<x1;x++)
      {
        mask(x,y) = V3uc(255.0f*imageBlob.rgbColor);
      }
    }
    texImage = GLTexture2D(mask);
  }
  else
  if (showBGSMask && view.mask.numel())
  {
    A2V3uc mask(view.image.size());
    #pragma omp parallel for
    for (int i=0;i<mask.numel();i++)
    {
      if (view.mask[i] == 0)
      {
        mask[i] = V3uc(0,0,0);
      }
      else
      {
        mask[i] = view.image[i];
      }
    }
    texImage = GLTexture2D(mask);
  }
  else
  {
    texImage = GLTexture2D(view.image);
  }
  texImage.setMagFilter(GL_NEAREST);
  texImage.setMinFilter(GL_LINEAR);       
  if (!hideVideo)
  {
    texImage.bind();
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_CULL_FACE);
    glBegin(GL_QUADS);
      glColor3f(1,1,1); glTexCoord2f(0,0); glVertex2f(0,0);
      glColor3f(1,1,1); glTexCoord2f(1,0); glVertex2f(view.image.width(),0);
      glColor3f(1,1,1); glTexCoord2f(1,1); glVertex2f(view.image.width(),view.image.height());
      glColor3f(1,1,1); glTexCoord2f(0,1); glVertex2f(0,view.image.height());
    glEnd();
    glDisable(GL_TEXTURE_2D);
  }

  glClear(GL_DEPTH_BUFFER_BIT);
#endif
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(topLeft(0),bottomRight(0),bottomRight(1),topLeft(1), n, f);
  glMultMatrixf(transpose(Mproj).data());

  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf(transpose(Mview).data());

  glEnable(GL_DEPTH_TEST);
  //glClear(GL_DEPTH_BUFFER_BIT);
#ifdef USE_OPENGL
  if (showCameras)
  {
    for (int i=0;i<views.size();i++) { drawCamera(views[i].camera,views[i].image.width(),views[i].image.height(),2.6*2.0); }        
  }
#endif
  std::vector<V3f> normals = calcVertexNormals(deformedVertices,model.triangles);

#ifdef USE_OPENGL

  static int lastFrame = -1;
  

  if (showBlobModel)
  {
    std::vector<V3f> blobCenters = blobModel.deformBlobCenters(model, trsaAnim[frame]);
    //V3f lightDir = normalize(view.camera.C-blobCenters[0])+0.5f*cameraUp-0.7f*cross(cameraDirection,cameraUp);
    V3f lightDir = normalize(view.camera.C-blobCenters[0])+0.5f*V3f(0,1,0)-0.7f*V3f(1,0,0);
    for (int i=0;i<blobCenters.size();i++)
    {
      V3f c = blobModel.colors[i];

      if (modelViewMode == MODEL_VIEW_MODE_WEIGHTS)
      {
        c = V3f(0,0,0);
        for (int j=0;j<blobModel.blobWeights.width();j++)
        {
          c += blobModel.blobWeights(j,i)*model.JColors[j];
        }
      }

      drawSphere(blobCenters[i], blobModel.blobs[i].sigma, c, lightDir);
    }
  }

  if (hideVideo)
  {
    // Draw Axis
    glLineWidth(1);
    glBegin(GL_LINES);
      glColor3f(1.0f,0.0f,0.0f);
      glVertex3f(0.0f,0.0f,0.0f);
      glVertex3f(1.0f,0.0f,0.0f);

      glColor3f(0.0f,1.0f,0.0f);
      glVertex3f(0.0f,0.0f,0.0f);
      glVertex3f(0.0f,1.0f,0.0f);
      
      glColor3f(0.0f,0.0f,1.0f);
      glVertex3f(0.0f,0.0f,0.0f);
      glVertex3f(0.0f,0.0f,1.0f);
    glEnd();

    // Draw Grid
    glColor3f(0.4,0.4,0.4);
    glLineWidth(1);
    glBegin(GL_LINES);
    float k = 0.25f;
    int r = 8;
    float y = 0.0f;
    for (int i=-r;i<=+r;i++)
    {
      glVertex3f(-float(r)*k,y,float(i)*k);
      glVertex3f(+float(r)*k,y,float(i)*k);

      glVertex3f(float(i)*k,y,-float(r)*k);
      glVertex3f(float(i)*k,y,+float(r)*k);
    }
    glEnd();
  }

  if (showMesh)
  {
    glBegin(GL_TRIANGLES);
    {
      const std::vector<V3i>& triangles = model.triangles;
      for (int j=0;j<triangles.size();j++)
      {
        for (int i=0;i<3;i++)
        {
          const int vId = triangles[j][i];
          const V3f& v = deformedVertices[vId];
          const V3f& n = normals[vId];
          
          if (j==hitId)
          {
            glColor3f(1,0,0);
          }
          else
          {
            if (modelViewMode == MODEL_VIEW_MODE_NORMALS)
            {
              V3f c = (n+V3f(1,1,1))*0.5f;
              glColor4f(c[0],c[1],c[2],0.5f);
            }
            else if (modelViewMode == MODEL_VIEW_MODE_COLORS)
            {
              const V3f& c = model.colors[vId];
              glColor(c);
            }
            else if (modelViewMode == MODEL_VIEW_MODE_WEIGHTS)
            {
              glColor(model.skinningWeightColors[vId]);
            }
            else
            {
              glColor3f(0.0f,0.0f,0.0f);
            }
            if (vertexSelection[vId])
            {
              glColor4f(1.0f,0.0f,0.0f,0.5f);
            }
          }
          glVertex3f(v[0],v[1],v[2]);
        }
      }
    }
    glEnd();
  }
  glDisable(GL_BLEND);

  glDisable(GL_DEPTH_TEST);

  if (showJoints)
  {
    glLineWidth(1);
    glBegin(GL_LINES);
      glColor3f(1.0f,1.0f,0.0f);
      for (int i=1;i<model.joints.size();i++)
      {
        glVertex(deformedJoints[i]);
        glVertex(deformedJoints[model.joints[i].parentId]);
      }
    glEnd();
  }

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(topLeft(0),bottomRight(0),bottomRight(1),topLeft(1));             
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Draw Anchors
  {
    std::vector<Mat4x4f> Ms(model.joints.size());

    for (int j=0;j<model.joints.size();j++)
    {
      const Mat4x4f M0 = evalJointTransform(j,model.joints,orgAngles);
      const Mat4x4f M1 = evalJointTransform(j,model.joints,curAngles);
      Ms[j] = M1*Mat4x4f(inverse(M0));
    }
     
    glDisable(GL_DEPTH_TEST);
    for (int i=0;i<anchors.size();i++)
    {
      const Anchor& anchor = anchors[i];
      if (anchor.viewId==selView)
      {
        V3f dv = V3f(0,0,0);
        for (int j=0;j<model.joints.size();j++)
        {
          float w = anchor.jointWeights[j];
          if (w>0)
          {
            V3f dvPart = p2e(Ms[j]*e2p(anchor.meshPoint));
            dv += w*dvPart;
          }
        }

        dv = p2e(M*e2p(dv));
        glPointSize(6);
        glBegin(GL_POINTS);
        glColor3f(1,0,0);
        glVertex(p2e(view.camera.P*e2p(dv)));
        glColor3f(0,0,1);
        glVertex(anchor.viewPoint);
        glEnd();

        glBegin(GL_LINES);
        glColor3f(1,0,0);
        glVertex(p2e(view.camera.P*e2p(dv)));
        glColor3f(0,0,1);
        glVertex(anchor.viewPoint);
        glEnd();
      }
    }

    if (bboxSelector.viewId == selView)
    {
      glLineWidth(2);
      glBegin(GL_LINE_LOOP);
      glColor3f(0,1,0);
      glVertex2f(bboxSelector.bboxMin[0], bboxSelector.bboxMin[1]);
      glVertex2f(bboxSelector.bboxMax[0], bboxSelector.bboxMin[1]);
      glVertex2f(bboxSelector.bboxMax[0], bboxSelector.bboxMax[1]);
      glVertex2f(bboxSelector.bboxMin[0], bboxSelector.bboxMax[1]);
      glEnd();
      glLineWidth(1);
    }
  }

#endif
  if (anchors.size())
  {
    if (optimizeScale)
    {
      deparametrizeTsAndAngles(minimizeLBFGS(AlignEnergy(views,anchors,anchorSamples.points,anchorSamples.weights,fixedIndices,fixedDeformedAnchorPoints,model.joints,prevAngles,angleLambda),
        parametrizeTsAndAngles(trsaAnim[frame]),5),&trsaAnim[frame]);
    }
    else
    {
      deparametrizeTAndAngles(minimizeLBFGS(AlignEnergy(views,anchors,anchorSamples.points,anchorSamples.weights,fixedIndices,fixedDeformedAnchorPoints,model.joints,prevAngles,angleLambda,trsaAnim[frame].trs.s),
        parametrizeTAndAngles(trsaAnim[frame]),5),&trsaAnim[frame]);
    }
    //trs.r = V3f(0,0,0);
  }

  if (fitModel)
  {
    double t0 = timerGet();
    deparametrizeTAndAngles(minimizeLBFGS(TrackBlobsEnergy(views, blobModel, model.joints, trsaAnim[frame].trs),
                            parametrizeTAndAngles(trsaAnim[frame]),optIters),&trsaAnim[frame]);
    printf("fitModel: %fs\n", (timerGet()-t0)*0.001);

    fitModel = false;
  }

  if (track != 0)
  {
    double t0 = timerGet();
    deparametrizeTAndAngles(minimizeLBFGS(TrackBlobsEnergy(views, blobModel, model.joints, trsaAnim[frame].trs),
                            parametrizeTAndAngles(trsaAnim[frame]),optIters),&trsaAnim[frame]);
    printf("TrackBlobsEnergy = %.3fs [frame: %d]\n",(timerGet()-t0)*0.001,frame);

    predicatePose(&trsaAnim, frame+track, track, predVelocity);
  }
}


int main(int argc,char** argv)
{
  if (argc<9)
  {
    printf("usage:\n");
    printf("\t%s dataPath %%s/cam%%d.PKRC first last %%s/cam%%d.mp4 %%s/bkg%%d.png model.skin anim.trsa\n", argv[0]);
    return 1;
  }

  const char* dataPath = argv[1];
  const char* cameraFileFormat = argv[2];
  const int cameraFirst = atoi(argv[3]);
  const int cameraLast = atoi(argv[4]);
  const char* videoFileFormat = argv[5];
  const char* backgroundFileFormat = argv[6];
  const char* skinningModelFileName = argv[7];
  const char* trsaFileName = argv[8];

  int lastFrame = -1;
  int frame = 0;
  
  timerReset();

  // omp_set_dynamic(0);
  // omp_set_num_threads(1);

  initGetMaskLUT();

  _chdir(dataPath);

  const int numCameras = (cameraLast-cameraFirst+1);
  std::vector<Video> videos(numCameras);
  std::vector<Camera> cameras(numCameras);
  std::vector<View> views(numCameras);
  bgs.resize(numCameras);
  
  bboxSelector.viewId = -1;

  #pragma omp parallel for
  for (int i=0;i<numCameras;i++)
  {
    Video& video = videos[i];
    Camera& camera = cameras[i];

    const std::string videoFileName  = spf(videoFileFormat,cameraFirst+i);
    const std::string cameraFileName = spf(cameraFileFormat,cameraFirst+i);
    const std::string backgroundFileName = spf(backgroundFileFormat,cameraFirst+i);

    if (!imread(&(bgs[i]), backgroundFileName)) { printf("ERROR opening %s\n", backgroundFileName.c_str()); exit(0); }

    video = vidOpen(videoFileName.c_str());
    if (!video) { printf("ERROR opening %s\n",videoFileName.c_str()); exit(0); }
    
    if (!readCamera(cameraFileName.c_str(),&camera)) { printf("ERROR opening %s\n",cameraFileName.c_str()); exit(0); }

    {
      Eigen::Matrix<double,3,4> P;
      for (int i=0;i<3;i++)
      for (int j=0;j<4;j++)
      {
        P(i,j) = camera.P(i,j);
      }
      
      Eigen::Matrix3d R;
      Eigen::Matrix3d K;
      Eigen::Vector3d C;
      Eigen::Vector3d t;
      
      decomposePMatrix2(P,R,K,C,t);

      for (int i=0;i<3;i++)
      for (int j=0;j<3;j++)
      {
        camera.R(i,j) = R(i,j);
        camera.K(i,j) = K(i,j);
      }
      for (int i=0;i<3;i++)
      {
        camera.C(i) = C(i);
      }
    }
  }

  int frameCount = INT_MAX;
  for (int i=0;i<numCameras;i++)
  {
    frameCount = std::min(frameCount,vidGetNumFrames(videos[i]));
  }
  printf("frameCount = %d\n",frameCount);

  SkinningModel model;
  model.load(skinningModelFileName);
  
  initJointLimits(model, &jointMinLimits, &jointMaxLimits);  

  blobModel.build(model);
  
  vertexSelection = std::vector<bool>(model.vertices.size(), false);

  genSamples(model.vertices, model.triangles, model.weights, calcVertexNormals(model.vertices, model.triangles), 0.03f, &anchorSamples);

  if (!readTRSA(&trsaAnim, trsaFileName))
  {
    TRS<float> trs;
    trs.t = V3f(0,0,0);
    trs.r = V3f(0,0,0);
    trs.s = 1.0f;

    trsaAnim.resize(frameCount);
    for (int i=0;i<frameCount;i++)
    {
      trsaAnim[i].trs = trs;
      trsaAnim[i].angles = std::vector<V3f>(model.joints.size(), V3f(0,0,0));
    }
  }

  std::vector<View2D> view2ds(numCameras);
  for (int i=0;i<numCameras;i++)
  {
    View2D& view2d = view2ds[i];
    view2d.width = -1;
    view2d.height = -1;
    view2d.panX = 0;
    view2d.panY = 0;
    view2d.zoom = 1;
    view2d.angle = 0;
  }
  int selView = 3;
  
  guiInit();
  GLContext* glCtx = new GLContext();
  glCtx->makeCurrent();
  glewInit();

  double t0;
  double t1 = timerGet();
  while (1)
  {
    prevAngles = trsaAnim[frame].angles;
    
    if (frame!=lastFrame)
    {
      anchors.clear();
      loadViews(videos, cameras, frame, bgs, trsaAnim, model, blobModel, bgsThr, imageBlobSize, &views);
      lastFrame = frame;
    }
    WindowBegin(ID,"",spf("Kostilam [%s]",trsaFileName).c_str(),Opts().showMaximized(true));

    if (windowCloseRequest()||keyDown(KeyEscape)) return false;

    HBoxLayoutBegin(ID);
      VBoxLayoutBegin(ID);
        HBoxLayoutBegin(ID);
          GLWidgetBegin(ID,glCtx,Opts().minimumWidth(800));
            doView(frame,numCameras,views,&view2ds,&selView,model,videos);
          GLWidgetEnd();
        HBoxLayoutEnd();

        HBoxLayoutBegin(ID);
          Label(ID,spf("Frame: %d", frame).c_str(),Opts().fixedWidth(64));
          HScrollBar(ID,0,frameCount-1,&frame,Opts().pageStep((frameCount)/10));
        HBoxLayoutEnd();
      VBoxLayoutEnd();

      FrameBegin(ID,Opts().fixedWidth(160));
        VBoxLayoutBegin(ID);
          if (Button(ID,"Clear Anchors"))
          {
            anchors.clear();
            prevAngles = trsaAnim[frame].angles;
            bboxSelector.viewId = -1;
            fixedIndices.clear();
            fixedDeformedAnchorPoints.clear();
            #pragma omp parallel for
            for (int i=0;i<vertexSelection.size();i++)
            {
              vertexSelection[i] = false;
            }
          }
          HSeparator(ID);
          CheckBox(ID, "Show Cameras", &showCameras);
          CheckBox(ID, "Hide Video", &hideVideo);
          CheckBox(ID, "Show Mesh", &showMesh);
          CheckBox(ID, "Show Joints", &showJoints);
          CheckBox(ID, "Show Blob Model", &showBlobModel);
          CheckBox(ID, "Show BGS Mask", &showBGSMask);
          CheckBox(ID, "Show Image Blobs", &showImageBlobs);

          // HBoxLayoutBegin(ID);
          //   Label(ID, "Angle Lambda:");
          //   SpinBox(ID, 0.0f, 1000000.0f, &angleLambda);
          // HBoxLayoutEnd();
          // HBoxLayoutBegin(ID);
          //   Label(ID, "Blob Angle Lambda:");
          //   SpinBox(ID, 0.0f, 10.0f, &blobAngleLambda);
          // HBoxLayoutEnd();

          Label(ID, "Model View Mode:");
          RadioButton(ID, "Normals", MODEL_VIEW_MODE_NORMALS, (int*)&modelViewMode);
          RadioButton(ID, "Colors",  MODEL_VIEW_MODE_COLORS,  (int*)&modelViewMode);
          RadioButton(ID, "Skinning Weights", MODEL_VIEW_MODE_WEIGHTS, (int*)&modelViewMode);
          
          if (track != 0)
          {
            if ((frame >= 0) && (frame < frameCount))
            {
              frame += track;
              if (frame == frameCount)
              {
                frame = frameCount-1;
                track = 0;
              }
              if (frame < 0)
              {
                frame = 0;
                track = 0;
              }
            }
            else
            {
              track = 0;
            }
          }
          CheckBox(ID, "Optimize Scale", &optimizeScale);
          Label(ID, "Optimization Iterations:");
          SpinBox(ID, 1, 10000, &optIters);
          Label(ID, "Pose Prediction Velocity:");
          SpinBox(ID, 0.0f, 1.0f, &predVelocity);
          Label(ID, "Background Subtraction\nThreshold:");
          if (SpinBox(ID, 0.0f, 255.0f, &bgsThr))
          {
            loadViews(videos, cameras, frame, bgs, trsaAnim, model, blobModel, bgsThr, imageBlobSize, &views);
          }
          Label(ID, "Image Blob Size:");
          if (SpinBox(ID, 2, 64, &imageBlobSize))
          {
            loadViews(videos, cameras, frame, bgs, trsaAnim, model, blobModel, bgsThr, imageBlobSize, &views);
          }
          if (Button(ID, "Train Colors"))
          {
            blobModel.updateBlobColors(views, model, trsaAnim[frame]);
          }
          if (track != 0)
          {
            if (Button(ID, "Stop Tracking"))
            {
              track = 0;
            }
          }
          else
          {
            HBoxLayoutBegin(ID);
              if (Button(ID, "<< Track"))
              {
                track = -1;
              }
              if (Button(ID, "Track >>"))
              {
                track = 1;
              }
            HBoxLayoutEnd();
          }
          if (Button(ID, "Fit Model Pose"))
          {
            fitModel = true;
          }
          if (Button(ID, "Reset Pose"))
          {
            trsaAnim[frame].trs.t = V3f(0,0,0);
            trsaAnim[frame].trs.r = V3f(0,0,0);
            trsaAnim[frame].angles = std::vector<V3f>(trsaAnim[frame].angles.size(),V3f(0,0,0));
          }
          ToggleButton(ID, playback ? "Stop Playback" : "Start Playback", &playback);
          if (playback)
          {
            frame = (frame + 1)%frameCount;
          }
          if (Button(ID,"Save Animation"))
          {
            if (!writeTRSAAnim(trsaAnim, trsaFileName))
            {
              printf("%s was not stored!\n", trsaFileName);
            }
          }
          if (Button(ID, "Export Animation"))
          {
            std::vector<FbxCameraParams> camerasParams(views.size());
            for (int i=0;i<views.size();i++)
            {
              const View& view = views[i];
              int w = view.image.width();
              int h = view.image.height();
              camerasParams[i].name = spf("KostilamCam%d",i);
              camerasParams[i].location = view.camera.C;
              camerasParams[i].interestPosition = view.camera.C + unproject(view.camera,V2f(w/2,h/2));
              camerasParams[i].apertureWidth  = w / 1000.0f;
              camerasParams[i].apertureHeight = h / 1000.0f;
            }
            exportAnim(model, trsaAnim, camerasParams, "c.fbx");
          }
          
          Spacer(ID);
        VBoxLayoutEnd();
      FrameEnd();
    HBoxLayoutEnd();

    WindowEnd();
    guiUpdate();  
  }

  return 0;
}
