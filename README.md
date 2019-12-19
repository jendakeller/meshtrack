# Kostilam
The application for performance capture.

## Building
Build the application just by running build.bat using Visual Studio.

## Basic usage
```
kostilam.exe dataPath %s/cam%d.pkrc first last %s/cam%d.mp4 %s/bkg%d.png model.skin anim.sta
```

### Shortcuts
**C** - show/hide cameras  
**M** - show/hide skinned mesh model  
**J** - show/hide joints  
**B** - show/hide blob model  
**I** - show/hide image blobs  
**F + left mouse button** - select mesh area to be fixed  
**F + Shift + left mouse button** - select additional mesh area to be fixed  
**left mouse button** - add anchor  
**right mouse button** - remove last added anchor  
**Ctrl + left mouse button** - moves cameras/ground along vertical axis  
**Ctrl + right mouse button** - changes scale of cameras  
**left/right arrow key** - switches individual views  
**Esc** - quits the application  

### GUI controls
**Clear Anchors** - removes all anchors  

**Show Cameras** - show/hide cameras  
**Hide Video** - show/hide video frame  
**Show Mesh** - show/hide skinned mesh model  
**Show Joints** - show/hide joints  
**Show Blob Model** - show/hide blob model  
**Show BGS Mask** - show/hide background subtraction mask  
**Show Image Blobs** - show/hide image blobs  

**Model View Mode:**  
**Normals** - visualize normal map  
**Colors** - visualize trained colors  
**Skinning Weights** - visualize skinning weights  

**Optimize Scale** - optimize model's scale during alignment  

**Optimization Iterations** - maximal number of iterations of optimization solver used for tracking  
**Pose Prediction Velocity** - when the velocity is zero, pose from the current frame is copied to the following frame, so no pose prediction is used. When the velocity is one pose is linearly interpolated from the preceding and the current frame to the following frame  
**Background Subtraction Threshold** - increases/decreases number of image blobs  
**Image Blob Size** - changes image blob size. The bigger size is the faster and less precise is tracking  
**Opacity** - changes mesh model's opacity  

**Train Colors** - train blob model's colors and skinned mesh model's colors  
**Track** - starts forward/backward tracking  
**Fit Model Pose** - align model to the current frame  
**Reset Pose** - reset model to its rest pose  
**Start Playback** - start playback  
**Filter Animation** - gaussian smoothing of model position and individual joint deformations  
**Save Animation** - stores the animation to the .sta file specified on application startup  
**Export Animation** - stores animation to the selected .fbx file  
**Align Cameras** - align cameras coordinate system to the model's coordinate system  
**Save Cameras** - rewrite .pkrc files containing camera parameters  

There are other tools for data preprocessing. Please, check following repositories:  
https://github.com/jamriska/kostilam-gopro-undistort  
https://github.com/jamriska/kostilam-findbulb  
https://github.com/jamriska/kostilam-selfcal  
https://github.com/jendakeller/kostilam-skinner

## License
The code is released into the public domain. You can do anything you want with it.

However, you should be aware that the application is using [liblbfgs](https://github.com/chokkan/liblbfgs) library,
[FBX SDK](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2019-0) and SMPL[1] model.
It is your responsibility to make sure you're not infringing any patent holder's rights by using this code.

## References

1. SMPL: A Skinned Multi-Person Linear Model  
   Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.  
   ACM Transactions on Graphics 34, 6 (Proc. SIGGRAPH Asia) (2015), 248.
