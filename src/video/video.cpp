#include <cstdio>
#include <cmath>
#include <vector>

#include "md5.h"
#include "ffmpeg_fas.h"

#include "video.h"

static bool isFasInitialized = false;

static bool write_table_file(FILE* file,seek_table_type table)
{
  seek_entry_type *entry;
  int index;

  if (NULL == table.array || table.num_entries <= 0) return false;

  int completed_flag = 0;
  if (table.completed == seek_true)
    completed_flag = 1;

  fprintf(file, "%d %d %d\n", table.num_frames, table.num_entries, completed_flag);
  for (index = 0; index < table.num_entries; index++)
  {
    entry = &(table.array[index]);

    fprintf (file, "%d %lld %lld\n", entry->display_index, entry->first_packet_dts, entry->last_packet_dts);
  }

  return true;
}

static seek_table_type read_table_file(FILE* file)
{
  seek_table_type ans = { NULL, (seek_boolean_type) 0, (seek_boolean_type) 0 };

  FILE *table_file = file;
  if (table_file == NULL) { return ans; }

  int completed_flag;
  fscanf(table_file, "%d %d %d\n", &ans.num_frames, &ans.num_entries, &completed_flag);

  if (completed_flag == 1)
    ans.completed = seek_true;
  else
    ans.completed = seek_false;

  ans.allocated_size = ans.num_entries;
  ans.array = (seek_entry_type*) malloc (ans.allocated_size * sizeof(seek_entry_type));

  int i;
  for (i=0;i<ans.num_entries;i++)
    fscanf(table_file, "%d %lld %lld\n", &(ans.array[i].display_index), &(ans.array[i].first_packet_dts), &(ans.array[i].last_packet_dts));

  return ans;
}

static void hashBlock(FILE* file,const uint64_t offset,const int blockSize,md5_state_t* inout_md5state)
{
  md5_state_t& md5state = *inout_md5state;
  _fseeki64(file,offset,SEEK_SET);

  std::vector<unsigned char> block(blockSize,0);
  const size_t size = fread(block.data(),1,block.size(),file);
  if (size==0) { return; }

  md5_append(&md5state,block.data(),size);
}

static std::string md5BlockwiseDigest(const std::string& fileName)
{
  FILE* f = fopen(fileName.c_str(),"rb");

  if(!f) { return std::string(""); }

  _fseeki64(f,0,SEEK_END);
  const uint64_t fileSize = _ftelli64(f);
  _fseeki64(f,0,SEEK_SET);

  const int blockSize = 16*1024*1024;

  md5_state_t md5state;
  md5_init(&md5state);

  if (fileSize<3*blockSize)
  {
    hashBlock(f,0,fileSize,&md5state);
  }
  else
  {
    hashBlock(f,0                   ,blockSize,&md5state);
    hashBlock(f,fileSize/2          ,blockSize,&md5state);
    hashBlock(f,fileSize-blockSize-1,blockSize,&md5state);
  }

  fclose(f);

  md5_append(&md5state,(const md5_byte_t*)&fileSize,sizeof(fileSize));

  md5_byte_t digest[16];
  md5_finish(&md5state,digest);

  char hexOutput[16*2+1] = {'\0'};
  for(int i=0;i<16;i++) { sprintf(hexOutput+i*2,"%02x",digest[i]); }

  return std::string(hexOutput);
}

Video vidOpen(const char* videoFileName)
{
  if (!isFasInitialized) { fas_initialize(FAS_TRUE,FAS_RGB24); }

  fas_context_ref_type video;

  if (fas_open_video(&video,(char*)videoFileName)!=FAS_SUCCESS) { return 0; }

  const std::string md5 = md5BlockwiseDigest(videoFileName);

  const std::string seekTableFileName = std::string(videoFileName)+".skt";

  bool rebuildSeekTable = true;

  {
    FILE* f = 0;
    f = fopen(seekTableFileName.c_str(),"rb");
    if (!f) { goto bail; }

    {
      std::string md5stored(32,'0');
      for(int i=0;i<32;i++) { if (fscanf(f,"%c",&md5stored[i])!=1) { printf("failed to read stored md5\n"); goto bail; } }
      if (md5!=md5stored) { printf("stored md5 doesn't match!\n"); goto bail; }
    }

    seek_table_type seek_table = read_table_file(f);

    fas_put_seek_table(video,seek_table);
    seek_release_table(&seek_table);

    rebuildSeekTable = false;
    //printf("seek table found!\n");

    bail:
      if (f) { fclose(f); }
  }

  if (rebuildSeekTable)
  {
    printf("rebuilding seek table\n");

    fas_get_frame_count(video); // force seek table rebuild

    const seek_table_type seek_table = fas_get_seek_table(video);

    FILE* f = fopen(seekTableFileName.c_str(),"wb");
    if (f)
    {
      fprintf(f,"%s\n",md5.c_str());
      write_table_file(f,seek_table);
      fclose(f);
    }
  }

  //if (out_frameCount!=0) { *out_frameCount = fas_get_frame_count(video); }

  return (Video)video;
}

/*
void getFrameSize(Video video,int* out_width,int* out_height)
{
  if (vidGetNumFrames(video)>0)
  {
    fas_raw_image_type image_buffer;
    fas_seek_to_frame((fas_context_ref_type)video,0);
    if (fas_get_frame((fas_context_ref_type)video,&image_buffer)!=FAS_SUCCESS) { return; }
    fas_free_frame(image_buffer);

    if (out_width!=0)  { *out_width = image_buffer.width; }
    if (out_height!=0) { *out_height = image_buffer.height; }
  }
  else
  {
    if (out_width!=0)  { *out_width = 0; }
    if (out_height!=0) { *out_height = 0; }
  }
}
*/

int vidGetWidth(Video video)
{
  //int width; getFrameSize(video,&width,0); return width;
  return fas_get_width((fas_context_ref_type)video);
}

int vidGetHeight(Video video)
{
  //int height; getFrameSize(video,0,&height); return height;
  return fas_get_height((fas_context_ref_type)video);
}

int vidGetNumFrames(Video video)
{
  return (video!=0) ? fas_get_frame_count((fas_context_ref_type)video) : 0;
}

void vidGetFrameData(Video video,int frame,void* data)
{
  if (video!=0 && frame>=0 && frame<vidGetNumFrames(video))
  {
    fas_seek_to_frame((fas_context_ref_type)video,frame);

    if (fas_get_frame2((fas_context_ref_type)video,data)!=FAS_SUCCESS) { return; }
  }
}

void vidClose(Video video)
{
  fas_close_video((fas_context_ref_type)video);
}

float vidGetFps(Video video)
{
  return (float)(1.0/(fas_get_frame_duration((fas_context_ref_type)video)/10000000.0));
}
