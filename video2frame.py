import sys
import os
import argparse

# parser = argparse.ArgumentParser(description='Video Anomaly Detection')
# parser.add_argument('--n', default='/media/yaman/new-e/Major-Project/VIS/video/boxer.mp4', type=str, help='file name')
# args = parser.parse_args()

# file_name =args.n
# # file_name ='/media/yaman/new-e/Major-Project/VIS/video/Abuse001_x264.mp4'


def to_frames(url, vid):
    # print(vid[:-4])/media/yaman/new-e/Django-Anomaly-Detection/media/input.mp4
    file_name = "media/"+vid
    if not os.path.isdir(file_name[:-4]):
        # print(file_name[:-4])
        os.mkdir(file_name[:-4])

    save_name = '' + file_name[:-4] + '\\%05d.jpg'
    print('+++***', save_name)

    os.system('ffmpeg -i %s -r 25 -q:v 2 -vf scale=320:240 %s' %
              (file_name, save_name))
