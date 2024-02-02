import os 
from moviepy.editor import *
import cv2

"""
There is a bug with moviepy library. The chopped videos have incorrect meta-data information (2 min video has duration info as 45 mins)
So the audio is saved for 45 mins instead of 2 mins (43 mins of dummy values)
"""

target_folder = "/home/DSO_SSD/ywy/Youtube_voice/"

completed_files = os.listdir(target_folder)

corrupted_files = []


for f_name in os.listdir("/home/DSO_SSD/Asian_Speaker/asian_speaker_15k"):

    f_path = os.path.join("/home/DSO_SSD/Asian_Speaker/asian_speaker_15k",f_name)
    file_name = f_name.split('.')[0]
    save_name = target_folder + file_name + '.wav'

        # print(f_path, file_name)
        # print(save_name,"\n")

    if  file_name + '.wav' in completed_files:
            continue
    else:
        try:
                # moviepy has a bug
                # instead use opencv to estimate the true duration
                cap = cv2.VideoCapture(f_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = round(frame_count/fps,2)

                # now save the video
                audio = AudioFileClip(f_path)
                audio.duration = audio.end = duration
                audio.write_audiofile(save_name)
        except:
                corrupted_files.append(file_name + '.wav')

    #break

print("\n\nCORRUPTED FILES\n\n")
for i in corrupted_files: print(i)
