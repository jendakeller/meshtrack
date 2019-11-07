#ifndef SKINNING_MODEL_H
#define SKINNING_MODEL_H

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

struct Mesh
{
  std::vector<V3i> triangles;
  std::vector<V3f> vertices;
  std::vector<V3f> normals;
  std::vector<V3f> colors;

  std::vector<V2f> texcoords;
  std::vector<V3i> triangleTexCoords;
};

std::vector<V3f> calcVertexNormals(const std::vector<V3f>& vertices,const std::vector<V3i>& triangles)
{
  std::vector<V3f> faceNormals(triangles.size());
  std::vector<V3f> vertexNormals(vertices.size(),V3f(0,0,0));
  
  for (int i=0;i<triangles.size();i++)
  {
    const V3i t = triangles[i];
    faceNormals[i] = normalize(cross(vertices[t[1]]-vertices[t[0]],
                                     vertices[t[2]]-vertices[t[0]]));
  }

  for (int i=0;i<triangles.size();i++)
  for (int j=0;j<3;j++)
  {
    vertexNormals[triangles[i][j]] += faceNormals[i];
  }
  for(int i=0;i<vertexNormals.size();i++)
  {
    vertexNormals[i] = normalize(vertexNormals[i]);
  }

  return vertexNormals;
}

Mesh loadMeshFromOBJ(const std::string& fileName)
{
  Mesh mesh;

  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string err;
  bool ret = tinyobj::LoadObj(&attrib,&shapes,&materials,&err,fileName.c_str(),"",true);
  
  if (!err.empty()) { printf("%s\n", err.c_str()); return mesh; }

  if (!ret) { return mesh; }

  int numTriangles = 0;
  for (int s=0;s<shapes.size();s++)
  {
    numTriangles += shapes[s].mesh.num_face_vertices.size();
  }
  mesh.triangles = std::vector<V3i>(numTriangles);

  int triangleIndex = 0;
  for (int s=0;s<shapes.size();s++)
  {
    for (int f=0;f<shapes[s].mesh.num_face_vertices.size();f++)
    {
      mesh.triangles[triangleIndex] = V3i(shapes[s].mesh.indices[3*f+0].vertex_index,
                                          shapes[s].mesh.indices[3*f+1].vertex_index,
                                          shapes[s].mesh.indices[3*f+2].vertex_index);
      triangleIndex++;
    }
  }

  const int numVertices = attrib.vertices.size()/3;
  mesh.vertices = std::vector<V3f>(numVertices);
  for (int i=0;i<numVertices;i++)
  {
    mesh.vertices[i] = V3f(attrib.vertices[i*3+0],
                           attrib.vertices[i*3+1],
                           attrib.vertices[i*3+2]);
  }

  int numTexCoords = attrib.texcoords.size() / 2;
  if (numTexCoords)
  {
    printf("loading texcoords\n");
    mesh.texcoords = std::vector<V2f>(numTexCoords);
    for (int i=0;i<numTexCoords;i++)
    {
      mesh.texcoords[i] = V2f(attrib.texcoords[i*2+0],
                              attrib.texcoords[i*2+1]);
    }

    mesh.triangleTexCoords = std::vector<V3i>(numTriangles);

    int triangleIndex = 0;
    for (int s=0;s<shapes.size();s++)
    {
      for (int f=0;f<shapes[s].mesh.num_face_vertices.size();f++)
      {
        mesh.triangleTexCoords[triangleIndex] = V3i(shapes[s].mesh.indices[3*f+0].texcoord_index,
                                                    shapes[s].mesh.indices[3*f+1].texcoord_index,
                                                    shapes[s].mesh.indices[3*f+2].texcoord_index);
        triangleIndex++;
      }
    }
  }

  mesh.normals = calcVertexNormals(mesh.vertices,mesh.triangles);
  mesh.colors = std::vector<V3f>(mesh.vertices.size(), V3f(0.5f,0.5f,0.5f));

  return mesh;
}

template<typename T>
struct Joint
{
  Vec<3,T> offset;
  int parentId;
};

class SkinningModel
{
  V3f evalJointPosition(const int jointId) const
  {
    const int parentId = joints[jointId].parentId;

    if (parentId >= 0)
    {
      return joints[jointId].offset + evalJointPosition(parentId);
    }
    else
    {
      return joints[jointId].offset;
    }
  }

public:
  std::string objFileName;
  std::vector<Joint<float>> joints;
  std::vector<V3f> vertices;
  std::vector<V3i> triangles;
  std::vector<V3f> normals;
  std::vector<V3f> colors;
  std::vector<V3f> skinningWeightColors;
  A2f weights;

  std::vector<V2f> texcoords;
  std::vector<V3i> triangleTexCoords;
  
  std::vector<V3f> JColors;

  void load(const char* filename)
  {
    FILE* fr = fopen(filename, "r");

    if (!fr) { return; }

    char strBuffer[256];
    fscanf(fr, "%s\n", strBuffer);
    objFileName = std::string(strBuffer);
    Mesh mesh = loadMeshFromOBJ(objFileName);
    vertices = mesh.vertices;
    triangles = mesh.triangles;
    colors = mesh.colors;

    texcoords = mesh.texcoords;
    triangleTexCoords = mesh.triangleTexCoords;

    int N = vertices.size();
    int numJoints;
    fscanf(fr, "%d\n", &numJoints);
    joints.resize(numJoints);
    for (int i=0;i<numJoints;i++)
    {
      V3f& offset = joints[i].offset;
      fscanf(fr, "%f %f %f %d\n", &(offset[0]), &(offset[1]), &(offset[2]), &(joints[i].parentId));
    }
    weights = A2f(numJoints, N);
    for (int i=0;i<N;i++)
    {
      for (int j=0;j<numJoints-1;j++)
      {
        fscanf(fr, "%f ", &(weights(j, i)));
      }
      fscanf(fr, "%f\n", &(weights(numJoints-1, i)));
    }

    #pragma omp parallel for
    for (int y=0;y<weights.height();y++)
    {
      float s = 0.0f;
      for (int x=0;x<weights.width();x++)
      {
        float v = std::min(std::max(weights(x,y), 0.0f), 1.0f);
        weights(x,y) = v;
        s += v;
      }
      for (int x=0;x<weights.width();x++)
      {
        weights(x,y) /= s;
      }
    }

    fclose(fr);

    normals = calcVertexNormals(vertices, triangles);
    // compute vertex colors
    {
      JColors.resize(numJoints);
      if (numJoints>0)  JColors[0] = V3f(1.0f, 0.0f, 0.0f); // kostrc
      if (numJoints>1)  JColors[1] = V3f(0.0f, 1.0f, 0.0f); // leva kycel
      if (numJoints>2)  JColors[2] = V3f(0.0f, 0.0f, 1.0f); // prava kycel
      if (numJoints>3)  JColors[3] = V3f(1.0f, 1.0f, 0.0f); // pupek
      if (numJoints>4)  JColors[4] = V3f(1.0f, 0.0f, 1.0f); // levy koleno
      if (numJoints>5)  JColors[5] = V3f(0.0f, 1.0f, 1.0f); // pravy koleno
      if (numJoints>6)  JColors[6] = V3f(0.5f, 0.5f, 0.5f); // nadpupek
      if (numJoints>7)  JColors[7] = V3f(0.5f, 1.0f, 0.0f); // levy kotnik
      if (numJoints>8)  JColors[8] = V3f(0.5f, 0.0f, 0.0f); // pravy kotnik
      if (numJoints>9)  JColors[9] = V3f(1.0f, 0.5f, 0.0f); // hrudnik
      if (numJoints>10) JColors[10] = V3f(0.5f, 0.125f, 0.125f); // leva spicka
      if (numJoints>11) JColors[11] = V3f(0.0f, 0.0f, 0.5f); // prava spicka
      if (numJoints>12) JColors[12] = V3f(0.0f, 0.5f, 0.0f); // krk
      if (numJoints>13) JColors[13] = V3f(0.0f, 0.5f, 0.5f); // leva lopatka
      if (numJoints>14) JColors[14] = V3f(0.5f, 0.5f, 0.0f); // prava lopatka
      if (numJoints>15) JColors[15] = V3f(0.75f, 0.75f, 0.75f); // hlava
      if (numJoints>16) JColors[16] = V3f(1.0f, 0.5f, 0.5f); // levy rameno
      if (numJoints>17) JColors[17] = V3f(0.9f, 0.6f, 0.125f); // pravy rameno
      if (numJoints>18) JColors[18] = V3f(0.25f, 0.0f, 0.5f); // levy loket
      if (numJoints>19) JColors[19] = V3f(0.5f, 0.0f, 0.25f); // pravy loket
      if (numJoints>20) JColors[20] = V3f(0.25f,0.75f,0.25f); // levy zapesti
      if (numJoints>21) JColors[21] = V3f(0.25f,0.25f,0.75f); // pravy zapesti
      if (numJoints>22) JColors[22] = V3f(0.75f,0.25f,0.25f); // konec levy ruky
      if (numJoints>23) JColors[23] = V3f(1.0f,1.0f,1.0f); // konec pravy ruky

      skinningWeightColors.resize(N);
      for (int i=0;i<N;i++)
      {
        V3f c(0,0,0);
        for (int j=0;j<numJoints;j++)
        {
          c += weights(j,i)*JColors[j];
        }
        skinningWeightColors[i] = c;
      }
    }
  }

  void save(const char* filename)
  {
    FILE* fw = fopen(filename, "w");

    if (!fw) { return; }

    fprintf(fw, "%s\n", objFileName.c_str());

    int N = vertices.size();
    
    int numJoints = weights.width();
    fprintf(fw, "%d\n", numJoints);
    for (int i=0;i<numJoints;i++)
    {
      V3f& offset = joints[i].offset;
      fprintf(fw, "%f %f %f %d\n", offset[0], offset[1], offset[2], joints[i].parentId);
    }
    for (int i=0;i<N;i++)
    {
      for (int j=0;j<numJoints-1;j++)
      {
        fprintf(fw, "%f ", weights(j, i));
      }
      fprintf(fw, "%f\n", weights(numJoints-1, i));
    }

    fclose(fw);
  }

  std::vector<V3f> getRestPoseJointPositions() const
  {
    std::vector<V3f> restPoseJointPositions(joints.size());

    for (int i=0;i<restPoseJointPositions.size();i++)
    {
      restPoseJointPositions[i] = evalJointPosition(i);
    }

    return restPoseJointPositions;
  }
};

#endif
