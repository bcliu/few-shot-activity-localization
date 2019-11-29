import glob, os
import cv2

os.mkdir("./new_frames/")

for split in ["ss", "fs"]:
    video_fns = glob.glob("new_videos/{}/*".format(split))

    os.mkdir("./new_frames/{}/".format(split))

    iter = 0

    for fn in video_fns:
        video_name = fn.split("/")[2] # 'new_videos/00054.mp4' to '00054.mp4'
        video_id = video_name.split(".")[0] # '00054.mp4' to '00054'

        os.mkdir("./new_frames/{}/{}/".format(split, video_id))

        vid = cv2.VideoCapture("{}".format(fn))
        success, image = vid.read()
        count = 0
        while success:
            cv2.imwrite("./new_frames/{}/{}/{:05d}.jpg".format(split, video_id, count), image)
            success, image = vid.read()
            count += 1

        print("Extracted {} frames from {}".format(count, fn))
        if iter % 100 == 0:
            print("{} / {} ({}%)".format(iter, len(video_fns), round(iter / len(video_fns) * 100.0, 2)))
        iter += 1
