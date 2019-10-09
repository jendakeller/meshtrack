#ifndef ANIMATION_H
#define ANIMATION_H

#include <fbxsdk.h>

#include "skinningModel.h"


#ifdef IOS_REF
  #undef  IOS_REF
  #define IOS_REF (*(pManager->GetIOSettings()))
#endif


template<typename T>
struct TRS
{
  Vec<3,T> t; // translation
  Vec<3,T> r; // rotation angles (xyz)
  T        s; // scale
};

template<typename T>
struct TRSA
{
  TRS<T> trs;
  std::vector<Vec<3,T>> angles;
};
typedef std::vector<TRSA<float>> TRSAAnim;

struct FbxCameraParams
{
  std::string name;
  V3f location;
  V3f interestPosition;
  float apertureWidth;
  float apertureHeight;
};

bool writeTRSAAnim(const TRSAAnim& trsaAnim, const std::string& fileName)
{
  FILE *fw = fopen(fileName.c_str(),"wb");
  if (!fw) { goto bail; }

  fprintf(fw,"%d\n",int(trsaAnim.size()));

  for (int i=0;i<trsaAnim.size();i++)
  {
    const TRSA<float>& trsa = trsaAnim[i];

    fprintf(fw,"%e\n",trsa.trs.t(0));
    fprintf(fw,"%e\n",trsa.trs.t(1));
    fprintf(fw,"%e\n",trsa.trs.t(2));

    fprintf(fw,"%e\n",trsa.trs.r(0));
    fprintf(fw,"%e\n",trsa.trs.r(1));
    fprintf(fw,"%e\n",trsa.trs.r(2));

    fprintf(fw,"%e\n",trsa.trs.s);

    fprintf(fw,"%d\n",int(trsa.angles.size()));
    for(int j=0;j<trsa.angles.size();j++)
    {
      const V3f& a = trsa.angles[j];
      fprintf(fw,"%e %e %e\n",a(0),a(1),a(2));  
    }
  }

  fclose(fw);
  return true;

bail:
  if (fw) { fclose(fw); }
  return false;
}

bool readTRSA(TRSAAnim* out_trsaAnim,const std::string& fileName)
{
  TRSAAnim trsaAnim;

  FILE* fr = fopen(fileName.c_str(),"rb");
  if (!fr) { goto bail; }
  
  int numFrames;
  if (fscanf(fr,"%d\n",&numFrames)!=1) { goto bail; }

  trsaAnim.resize(numFrames);

  for (int i=0;i<numFrames;i++)
  {
    TRSA<float>& trsa = trsaAnim[i];

    if (fscanf(fr,"%e\n",&trsa.trs.t(0))!=1) { goto bail; }
    if (fscanf(fr,"%e\n",&trsa.trs.t(1))!=1) { goto bail; }
    if (fscanf(fr,"%e\n",&trsa.trs.t(2))!=1) { goto bail; }

    if (fscanf(fr,"%e\n",&trsa.trs.r(0))!=1) { goto bail; }
    if (fscanf(fr,"%e\n",&trsa.trs.r(1))!=1) { goto bail; }
    if (fscanf(fr,"%e\n",&trsa.trs.r(2))!=1) { goto bail; }

    if (fscanf(fr,"%e\n",&trsa.trs.s)!=1)    { goto bail; }

    int numAngles = 0;
    if (fscanf(fr,"%d\n",&numAngles)!=1) { goto bail; }

    trsa.angles = std::vector<V3f>(numAngles);
    for(int j=0;j<trsa.angles.size();j++)
    {
      V3f& a = trsa.angles[j];
      if (fscanf(fr,"%e %e %e\n",&a(0),&a(1),&a(2))!=3) { goto bail; }
    }
  }

  fclose(fr);
  if (out_trsaAnim) { *out_trsaAnim = trsaAnim; }
  return true;

bail:
  if (fr) { fclose(fr); }
  return false;
}

void initializeSdkObjects(FbxManager*& pManager, FbxScene*& pScene)
{
  //The first thing to do is to create the FBX Manager which is the object allocator for almost all the classes in the SDK
  pManager = FbxManager::Create();
  if (!pManager)
  {
    FBXSDK_printf("Error: Unable to create FBX Manager!\n");
    exit(1);
  }
  else
  {
    FBXSDK_printf("Autodesk FBX SDK version %s\n", pManager->GetVersion());
  }

  //Create an IOSettings object. This object holds all import/export settings.
  FbxIOSettings* ios = FbxIOSettings::Create(pManager, IOSROOT);
  pManager->SetIOSettings(ios);

  //Load plugins from the executable directory (optional)
  FbxString lPath = FbxGetApplicationDirectory();
  pManager->LoadPluginsDirectory(lPath.Buffer());

  //Create an FBX scene. This object holds most objects imported/exported from/to files.
  pScene = FbxScene::Create(pManager, "Kostilam Scene");
  if (!pScene)
  {
    FBXSDK_printf("Error: Unable to create FBX scene!\n");
    exit(1);
  }
}

void destroySdkObjects(FbxManager* pManager, bool pExitStatus)
{
  //Delete the FBX Manager. All the objects that have been allocated using the FBX Manager and that haven't been explicitly destroyed are also automatically destroyed.
  if (pManager) pManager->Destroy();
  if (pExitStatus) FBXSDK_printf("Program Success!\n");
}

bool saveScene(FbxManager* pManager, FbxDocument* pScene, const char* pFilename, int pFileFormat=-1, bool pEmbedMedia=false)
{
  int lMajor, lMinor, lRevision;
  bool lStatus = true;

  // Create an exporter.
  FbxExporter* lExporter = FbxExporter::Create(pManager, "");

  if (pFileFormat < 0 || pFileFormat >= pManager->GetIOPluginRegistry()->GetWriterFormatCount())
  {
    // Write in fall back format in less no ASCII format found
    pFileFormat = pManager->GetIOPluginRegistry()->GetNativeWriterFormat();

    //Try to export in ASCII if possible
    int lFormatCount = pManager->GetIOPluginRegistry()->GetWriterFormatCount();

    for (int lFormatIndex=0;lFormatIndex<lFormatCount;lFormatIndex++)
    {
      if (pManager->GetIOPluginRegistry()->WriterIsFBX(lFormatIndex))
      {
        FbxString lDesc =pManager->GetIOPluginRegistry()->GetWriterFormatDescription(lFormatIndex);
        const char *lASCII = "ascii";
        if (lDesc.Find(lASCII)>=0)
        {
          pFileFormat = lFormatIndex;
          break;
        }
      }
    } 
  }

  // Set the export states. By default, the export states are always set to 
  // true except for the option eEXPORT_TEXTURE_AS_EMBEDDED. The code below 
  // shows how to change these states.
  IOS_REF.SetBoolProp(EXP_FBX_MATERIAL,        true);
  IOS_REF.SetBoolProp(EXP_FBX_TEXTURE,         true);
  IOS_REF.SetBoolProp(EXP_FBX_EMBEDDED,        pEmbedMedia);
  IOS_REF.SetBoolProp(EXP_FBX_SHAPE,           true);
  IOS_REF.SetBoolProp(EXP_FBX_GOBO,            true);
  IOS_REF.SetBoolProp(EXP_FBX_ANIMATION,       true);
  IOS_REF.SetBoolProp(EXP_FBX_GLOBAL_SETTINGS, true);

  // Initialize the exporter by providing a filename.
  if (lExporter->Initialize(pFilename, pFileFormat, pManager->GetIOSettings()) == false)
  {
    FBXSDK_printf("Call to FbxExporter::Initialize() failed.\n");
    FBXSDK_printf("Error returned: %s\n\n", lExporter->GetStatus().GetErrorString());
    return false;
  }

  FbxManager::GetFileFormatVersion(lMajor, lMinor, lRevision);
  FBXSDK_printf("FBX file format version %d.%d.%d\n\n", lMajor, lMinor, lRevision);

  // Export the scene.
  lStatus = lExporter->Export(pScene); 

  // Destroy the exporter.
  lExporter->Destroy();
  return lStatus;
}

bool loadScene(FbxManager* pManager, FbxDocument* pScene, const char* pFilename)
{
  int lFileMajor, lFileMinor, lFileRevision;
  int lSDKMajor,  lSDKMinor,  lSDKRevision;
  //int lFileFormat = -1;
  int lAnimStackCount;
  bool lStatus;
  char lPassword[1024];

  // Get the file version number generate by the FBX SDK.
  FbxManager::GetFileFormatVersion(lSDKMajor, lSDKMinor, lSDKRevision);

  // Create an importer.
  FbxImporter* lImporter = FbxImporter::Create(pManager,"");

  // Initialize the importer by providing a filename.
  const bool lImportStatus = lImporter->Initialize(pFilename, -1, pManager->GetIOSettings());
  lImporter->GetFileVersion(lFileMajor, lFileMinor, lFileRevision);

  if (!lImportStatus)
  {
    FbxString error = lImporter->GetStatus().GetErrorString();
    FBXSDK_printf("Call to FbxImporter::Initialize() failed.\n");
    FBXSDK_printf("Error returned: %s\n\n", error.Buffer());

    if (lImporter->GetStatus().GetCode() == FbxStatus::eInvalidFileVersion)
    {
      FBXSDK_printf("FBX file format version for this FBX SDK is %d.%d.%d\n", lSDKMajor, lSDKMinor, lSDKRevision);
      FBXSDK_printf("FBX file format version for file '%s' is %d.%d.%d\n\n", pFilename, lFileMajor, lFileMinor, lFileRevision);
    }

    return false;
  }

  FBXSDK_printf("FBX file format version for this FBX SDK is %d.%d.%d\n", lSDKMajor, lSDKMinor, lSDKRevision);

  if (lImporter->IsFBX())
  {
    FBXSDK_printf("FBX file format version for file '%s' is %d.%d.%d\n\n", pFilename, lFileMajor, lFileMinor, lFileRevision);

    // From this point, it is possible to access animation stack information without
    // the expense of loading the entire file.

    FBXSDK_printf("Animation Stack Information\n");

    lAnimStackCount = lImporter->GetAnimStackCount();

    FBXSDK_printf("    Number of Animation Stacks: %d\n", lAnimStackCount);
    FBXSDK_printf("    Current Animation Stack: \"%s\"\n", lImporter->GetActiveAnimStackName().Buffer());
    FBXSDK_printf("\n");

    for (int i=0;i<lAnimStackCount;i++)
    {
      FbxTakeInfo* lTakeInfo = lImporter->GetTakeInfo(i);

      FBXSDK_printf("    Animation Stack %d\n", i);
      FBXSDK_printf("         Name: \"%s\"\n", lTakeInfo->mName.Buffer());
      FBXSDK_printf("         Description: \"%s\"\n", lTakeInfo->mDescription.Buffer());

      // Change the value of the import name if the animation stack should be imported 
      // under a different name.
      FBXSDK_printf("         Import Name: \"%s\"\n", lTakeInfo->mImportName.Buffer());

      // Set the value of the import state to false if the animation stack should be not
      // be imported. 
      FBXSDK_printf("         Import State: %s\n", lTakeInfo->mSelect ? "true" : "false");
      FBXSDK_printf("\n");
    }

    // Set the import states. By default, the import states are always set to 
    // true. The code below shows how to change these states.
    IOS_REF.SetBoolProp(IMP_FBX_MATERIAL,        true);
    IOS_REF.SetBoolProp(IMP_FBX_TEXTURE,         true);
    IOS_REF.SetBoolProp(IMP_FBX_LINK,            true);
    IOS_REF.SetBoolProp(IMP_FBX_SHAPE,           true);
    IOS_REF.SetBoolProp(IMP_FBX_GOBO,            true);
    IOS_REF.SetBoolProp(IMP_FBX_ANIMATION,       true);
    IOS_REF.SetBoolProp(IMP_FBX_GLOBAL_SETTINGS, true);
  }

  // Import the scene.
  lStatus = lImporter->Import(pScene);

  if (lStatus == false && lImporter->GetStatus().GetCode() == FbxStatus::ePasswordError)
  {
    FBXSDK_printf("Please enter password: ");

    lPassword[0] = '\0';

    FBXSDK_CRT_SECURE_NO_WARNING_BEGIN
    scanf("%s", lPassword);
    FBXSDK_CRT_SECURE_NO_WARNING_END

    FbxString lString(lPassword);

    IOS_REF.SetStringProp(IMP_FBX_PASSWORD,      lString);
    IOS_REF.SetBoolProp(IMP_FBX_PASSWORD_ENABLE, true);

    lStatus = lImporter->Import(pScene);

    if (lStatus == false && lImporter->GetStatus().GetCode() == FbxStatus::ePasswordError)
    {
      FBXSDK_printf("\nPassword is wrong, import aborted.\n");
    }
  }

  // Destroy the importer.
  lImporter->Destroy();

  return lStatus;
}

FbxNode* createCamera(FbxScene* pScene, const FbxCameraParams& cameraParams)
{
  FbxCamera* lCamera = FbxCamera::Create(pScene,cameraParams.name.c_str());

  // Modify some camera default settings.
  lCamera->SetApertureMode(FbxCamera::eVertical);
  lCamera->SetApertureWidth(cameraParams.apertureWidth);
  lCamera->SetApertureHeight(cameraParams.apertureHeight);

  lCamera->InterestPosition.Set(FbxVector4(cameraParams.interestPosition[0],
                                           cameraParams.interestPosition[1],
                                           cameraParams.interestPosition[2]));

  FbxNode* lNode = FbxNode::Create(pScene,cameraParams.name.c_str());
  lNode->SetNodeAttribute(lCamera);

  lNode->LclTranslation.Set(FbxVector4(cameraParams.location[0],
                                       cameraParams.location[1],
                                       cameraParams.location[2]));
  
  return lNode;
}

FbxNode* createMesh(const SkinningModel& model, FbxScene* pScene, const char* pName)
{
  FbxMesh* lMesh = FbxMesh::Create(pScene,pName);
  
  lMesh->ReservePolygonCount(model.triangles.size());
  lMesh->ReservePolygonVertexCount(model.vertices.size());
  lMesh->InitControlPoints(model.vertices.size());
  
  FbxVector4* lVector4 = lMesh->GetControlPoints();
  for (int i=0;i<model.vertices.size();i++)
  {
    lVector4[i].Set(model.vertices[i][0],model.vertices[i][1],model.vertices[i][2]);
  }

  for (int i=0;i<model.triangles.size();i++)
  {
    lMesh->BeginPolygon();
    lMesh->AddPolygon(model.triangles[i][0]);
    lMesh->AddPolygon(model.triangles[i][1]);
    lMesh->AddPolygon(model.triangles[i][2]);
    lMesh->EndPolygon();
  }
  lMesh->BuildMeshEdgeArray();
  lMesh->GenerateNormals(false, true);

  FbxNode* lNode = FbxNode::Create(pScene,pName);
  lNode->SetNodeAttribute(lMesh);
  
  return lNode;
}

std::vector<FbxNode*> createSkeleton(const SkinningModel& model, FbxScene* pScene, const char* pName)
{
  std::vector<FbxString> jointNames(model.joints.size(), FbxString(pName));
  jointNames[0]  += "Pelvis";
  jointNames[1]  += "L_Hip";
  jointNames[2]  += "R_Hip";
  jointNames[3]  += "Spine1";
  jointNames[4]  += "L_Knee";
  jointNames[5]  += "R_Knee";
  jointNames[6]  += "Spine2";
  jointNames[7]  += "L_Ankle";
  jointNames[8]  += "R_Ankle";
  jointNames[9]  += "Spine3";
  jointNames[10] += "L_Foot";
  jointNames[11] += "R_Foot";
  jointNames[12] += "Neck";
  jointNames[13] += "L_Collar";
  jointNames[14] += "R_Collar";
  jointNames[15] += "Head";
  jointNames[16] += "L_Shoulder";
  jointNames[17] += "R_Shoulder";
  jointNames[18] += "L_Elbow";
  jointNames[19] += "R_Elbow";
  jointNames[20] += "L_Wrist";
  jointNames[21] += "R_Wrist";
  jointNames[22] += "L_Hand";
  jointNames[23] += "R_Hand";

  std::vector<FbxNode*> nodes(model.joints.size());
  
  for (int i=0;i<model.joints.size();i++)
  {
    FbxSkeleton* lSkeletonAttribute = FbxSkeleton::Create(pScene,jointNames[i]);
    lSkeletonAttribute->SetSkeletonType((i>0) ? FbxSkeleton::eLimbNode : FbxSkeleton::eRoot);
    lSkeletonAttribute->Size.Set(1.0);
    nodes[i] = FbxNode::Create(pScene,jointNames[i].Buffer());
    nodes[i]->SetNodeAttribute(lSkeletonAttribute);
    nodes[i]->LclTranslation.Set(FbxVector4(model.joints[i].offset[0], model.joints[i].offset[1], model.joints[i].offset[2]));
    
    if (i > 0)
    {
      nodes[model.joints[i].parentId]->AddChild(nodes[i]);
    }
  }

  return nodes;
}

void linkMeshToSkeleton(const SkinningModel& model, FbxScene* pScene, FbxNode* pMesh, std::vector<FbxNode*>& skeleton)
{
  FbxAMatrix lXMatrix;

  std::vector<FbxCluster*> clusters(skeleton.size());

  for (int i=0;i<skeleton.size();i++)
  {
    clusters[i] = FbxCluster::Create(pScene, "");
    clusters[i]->SetLink(skeleton[i]);
    clusters[i]->SetLinkMode(FbxCluster::eTotalOne);
    for (int y=0;y<model.weights.height();y++)
    {
      clusters[i]->AddControlPointIndex(y, model.weights(i, y));
    }
  }

  FbxScene* lScene = pMesh->GetScene();
  if( lScene ) lXMatrix = pMesh->EvaluateGlobalTransform();

  for (int i=0;i<skeleton.size();i++)
  {
    clusters[i]->SetTransformMatrix(lXMatrix);
  }

  for (int i=0;i<skeleton.size();i++)
  {
    if (lScene) lXMatrix = skeleton[i]->EvaluateGlobalTransform();
    clusters[i]->SetTransformLinkMatrix(lXMatrix);
  }

  FbxGeometry* lMeshAttribute = (FbxGeometry*) pMesh->GetNodeAttribute();
  FbxSkin* lSkin = FbxSkin::Create(pScene, "");

  for (int i=0;i<skeleton.size();i++)
  {
    lSkin->AddCluster(clusters[i]);
  }

  lMeshAttribute->AddDeformer(lSkin);
}

void animate(const SkinningModel& model, const TRSAAnim& trsaAnim, FbxScene* pScene, std::vector<FbxNode*>& skeleton)
{
  FbxString lAnimStackName;
  FbxTime lTime;
  int lKeyIndex = 0;

  lAnimStackName = "Kostilam";
  FbxAnimStack* lAnimStack = FbxAnimStack::Create(pScene, lAnimStackName);

  FbxAnimLayer* lAnimLayer = FbxAnimLayer::Create(pScene, "BaseLayer");
  lAnimStack->AddMember(lAnimLayer);

  std::vector<std::vector<FbxAnimCurve*>> rCurves(skeleton.size(),std::vector<FbxAnimCurve*>(3));

  for (int i=0;i<skeleton.size();i++)
  {
    rCurves[i][0] = skeleton[i]->LclRotation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
    rCurves[i][1] = skeleton[i]->LclRotation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
    rCurves[i][2] = skeleton[i]->LclRotation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);

    skeleton[i]->SetRotationActive(true);
    skeleton[i]->SetRotationOrder(FbxNode::eSourcePivot, FbxEuler::eOrderZXY);
  }
    
  std::vector<FbxAnimCurve*> tCurves(3);
  tCurves[0] = skeleton[0]->LclTranslation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
  tCurves[1] = skeleton[0]->LclTranslation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
  tCurves[2] = skeleton[0]->LclTranslation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);

  for (int i=0;i<trsaAnim.size();i++)
  {
    lTime.SetSecondDouble(i / 60.0);

    for (int j=0;j<skeleton.size();j++)
    for (int k=0;k<3;k++)
    {
      if (rCurves[j][k])
      {
        rCurves[j][k]->KeyModifyBegin();
        lKeyIndex = rCurves[j][k]->KeyAdd(lTime);
        float angle = trsaAnim[i].angles[j][k];
                
        rCurves[j][k]->KeySetValue(lKeyIndex, -angle/M_PI*180.0);
        rCurves[j][k]->KeySetInterpolation(lKeyIndex, FbxAnimCurveDef::eInterpolationConstant);
      }
    }

    for (int k=0;k<3;k++)
    {
      if (tCurves[k])
      {
        tCurves[k]->KeyModifyBegin();
        lKeyIndex = tCurves[k]->KeyAdd(lTime);
        tCurves[k]->KeySetValue(lKeyIndex, trsaAnim[i].trs.t[k] + model.joints[0].offset[k]);
        tCurves[k]->KeySetInterpolation(lKeyIndex, FbxAnimCurveDef::eInterpolationCubic);
      }
    }
  }
}

bool createScene(const SkinningModel& in_model, const TRSAAnim& trsaAnim, const std::vector<FbxCameraParams>& camerasParams, FbxManager* pSdkManager, FbxScene* pScene)
{
  // create scene info
  FbxDocumentInfo* sceneInfo = FbxDocumentInfo::Create(pSdkManager,"SceneInfo");
  sceneInfo->mTitle = "Kostilam";
  sceneInfo->mSubject = "Performance capture animation from Kostilam.";
  sceneInfo->mAuthor = "kostilam.exe";
  sceneInfo->mRevision = "rev. 1.0";
  sceneInfo->mKeywords = "performance capture";
  sceneInfo->mComment = "no particular comments required.";

  // we need to add the sceneInfo before calling AddThumbNailToScene because
  // that function is asking the scene for the sceneInfo.
  pScene->SetSceneInfo(sceneInfo);

  SkinningModel model = in_model;
  
  float scale = 1.0f;
  if (trsaAnim.size())
  {
    scale = trsaAnim[0].trs.s;
  }
  #pragma omp parallel for
  for (int i=0;i<model.vertices.size();i++)
  {
    model.vertices[i] *= scale;
  }
  for (int i=0;i<model.joints.size();i++)
  {
    model.joints[i].offset *= scale;
  }

  FbxNode* lMesh = createMesh(model, pScene, "Mesh");
  std::vector<FbxNode*> skeleton = createSkeleton(model, pScene, "SmplSkeleton");

  // Build the node tree.
  FbxNode* lRootNode = pScene->GetRootNode();
  lRootNode->AddChild(lMesh);
  lRootNode->AddChild(skeleton[0]);
  
  // Add cameras
  for (int i=0;i<camerasParams.size();i++)
  {
    FbxNode* camera = createCamera(pScene, camerasParams[i]);
    lRootNode->AddChild(camera);
    if (i==0)
    {
      pScene->GetGlobalSettings().SetDefaultCamera(camera->GetName());
    }
  }

  // Setup skinning weights
  linkMeshToSkeleton(model, pScene, lMesh, skeleton);
  
  // Animation
  animate(model, trsaAnim, pScene, skeleton);

  return true;
}

bool exportAnim(const SkinningModel& model, const TRSAAnim& trsaAnim, const std::vector<FbxCameraParams>& camerasParams, const std::string& fileName)
{
  FbxManager* lSdkManager = NULL;
  FbxScene* lScene = NULL;
  bool lResult;

  // Prepare the FBX SDK.
  initializeSdkObjects(lSdkManager, lScene);

  FbxGlobalSettings& lSettings = lScene->GetGlobalSettings();
  lSettings.SetTimeMode(FbxTime::eFrames60);
  
  // Create the scene.
  lResult = createScene(model, trsaAnim, camerasParams, lSdkManager, lScene);

  if (lResult == false)
  {
    FBXSDK_printf("\n\nAn error occurred while creating the scene...\n");
    destroySdkObjects(lSdkManager, lResult);
    return lResult;
  }

  // Save the scene.
  lResult = saveScene(lSdkManager, lScene, fileName.c_str());
  if (lResult == false)
  {
    FBXSDK_printf("\n\nAn error occurred while saving the scene...\n");
    destroySdkObjects(lSdkManager, lResult);
    return 0;
  }

  // Destroy all objects created by the FBX SDK.
  destroySdkObjects(lSdkManager, lResult);

  return lResult;
}

#endif
