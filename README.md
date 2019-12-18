# Kostilam
The application for performance capture.

## Building
Build the application just by running build.bat using Visual Studio.

## Basic usage
```
kostilam.exe dataPath %s/cam%d.pkrc first last %s/cam%d.mp4 %s/bkg%d.png model.skin anim.sta
```
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
