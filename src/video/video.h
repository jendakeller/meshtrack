#ifndef VIDEO_H_
#define VIDEO_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef void* Video;

Video vidOpen(const char* filename);
int   vidGetWidth(Video video);
int   vidGetHeight(Video video);
int   vidGetNumFrames(Video video);
void  vidGetFrameData(Video video,int frame,void* data);
void  vidClose(Video video);
float vidGetFps(Video video);

#ifdef __cplusplus
}
#endif

#endif
