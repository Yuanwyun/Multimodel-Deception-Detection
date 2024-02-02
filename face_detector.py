import cv2
from tools import FaceAlignmentTools
import os

print("\n\n\t\t\tPrecessing \n\n")

save_dir = "/home/DSO_SSD/ywy/Youtube_faceframes/"

tool = FaceAlignmentTools()



for f_name in os.listdir("/home/DSO_SSD/Asian_Speaker/asian_speaker_15k/"):
          
    f_path = os.path.join("/home/DSO_SSD/Asian_Speaker/asian_speaker_15k/",f_name)
    file_name = f_name.split('.')[0]

    num_frames = 0
    os.makedirs(save_dir + file_name, exist_ok=True)
    save_path = save_dir + file_name

    cap = cv2.VideoCapture(f_path)
    frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
    if frame_rate<20 or frame_rate>60:
        print("nooo")
    print(frame_rate, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    
        
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if ret == False: break

            # # save face-frames at 5 fps
            # if frame_rate == 30:
            #     if i%6 != 0: continue # only frames 0,3,6,9,... are saved. Other frames 1,2,4,5,7,8... are skipped!
            # if frame_rate == 60:
            #     if i%15 != 0: continue

            # save face-frames at 10 fps
        if 20 <= frame_rate <= 60:
            if i%10 != 0: continue
        

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aligned_img = tool.align(frame)

        if aligned_img is None: # if no face detected, do not increase the num-frames
            continue

        aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR)
        num_frames += 1

        s = "%04d" % num_frames
        save_image = save_path + '/frame_' + s + '.jpg'

        cv2.imwrite(save_image, aligned_img)

    #         break # frame level
    #     break # video level
    # break # subject level