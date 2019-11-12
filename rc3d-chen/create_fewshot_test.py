from os import listdir

from os.path import isfile, join, exists
import shutil

path = '/home/vltava/disk2/THUMOS14_fewshot/annotations_test'
files = [f for f in listdir(path) if isfile(join(path, f)) and join(path, f).find('detclasslist.txt') == -1]

for f in files:
    full_path = join(path, f)
    opened_f = open(full_path, 'r')
    lines = opened_f.readlines()
    opened_f.close()
    print(f)

    copied = 0
    for line in lines:
        video_name = line.split(' ')[0]
        src_file = '/home/vltava/disk2/THUMOS14/frames/test/' + video_name
        dest_file = '/home/vltava/disk2/THUMOS14_fewshot/frames/test/' + video_name
        if exists(dest_file):
            continue
        print(video_name)
        shutil.copytree(src_file, dest_file)
        copied += 1
        if copied >= 5:
            break
