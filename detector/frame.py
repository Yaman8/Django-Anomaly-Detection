import os

def extract_snippet(input,start,duration):
    start_hrs,start_min,start_sec=start.split(':')[0],start.split(':')[1],start.split(':')[2]
    duration_hrs,duration_min,duration_sec=duration.split(':')[0],duration.split(':')[1],duration.split(':')[2]
    print(start_hrs,start_min,start_sec)
    print(duration_hrs,duration_min,duration_sec)
    os.system('ffmpeg -y -i "%s" -ss "%s" -t "%s" -c copy "%s"'%(input,start_hrs+':'+start_min+':'+start_sec,duration_hrs+':'+duration_min+':'+duration_sec,input[:-4]+'_snippet.mp4'))
    # os.system('ffmpeg -i "%s" -c:v libx264 "%s"'%(save_path+'/%05d.jpg', save_path+'.mp4'))

# extract_snippet('/media/yaman/new-e/Django-Anomaly-Detection/media/video.mp4','00:00:07','00:00:10')