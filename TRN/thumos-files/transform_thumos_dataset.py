import os
import random

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

SPLITS = ["val", "test"]

def extract_subclips(annots, counter, output_folder):
    csv = []
    for i in range(len(annots)):
        video_name, start_time, end_time, label = annots[i]
        
        if "test" in video_name:
            ffmpeg_extract_subclip("./videos/test/{}.mp4".format(video_name), float(start_time), float(end_time), targetname = "./new_videos/{}/{:05d}.mp4".format(output_folder, counter))
        elif "validation" in video_name:
            ffmpeg_extract_subclip("./videos/val/{}.mp4".format(video_name), float(start_time), float(end_time), targetname = "./new_videos/{}/{:05d}.mp4".format(output_folder, counter))
        
        csv.append("{};{}".format(counter, label))
        counter += 1

        if i % 50 == 0:
            print("-- \t{} / {} ({}%)".format(i, len(annots), round(i / len(annots) * 100.0, 2)))
    return csv, counter

# read the classes from each strongly supervised and few-shot setup and strip newline from lines read in
print("- Opening strongly-supervised category.txt")
with open("thumos-ss-category.txt", "r") as f:
    category = f.readlines()
    category = [item.strip('\n') for item in category]
print("- Opening few-shot category.txt")
with open("thumos-fs-category.txt", "r") as f:
    fs_category = f.readlines()
    fs_category = [item.strip('\n') for item in fs_category]

# get all filenames for annotation files for strongly supervised and few-shot task
category_fns = []
fs_category_fns = []
print("- Creating category_fns and fs_category_fns")
for item in category:
    category_fns.append("annotations_val/{}_val.txt".format(item))
    category_fns.append("annotations_test/{}_test.txt".format(item))
for item in fs_category:
    fs_category_fns.append("annotations_val/{}_val.txt".format(item))
    fs_category_fns.append("annotations_test/{}_test.txt".format(item))

id_counter = 0

#os.mkdir("./new_videos")
#os.mkdir("./new_videos/ss/")
#os.mkdir("./new_videos/fs/")

##########################################
# STRONGLY SUPERVISED
##########################################
print("===== STRONGLY SUPERVISED =====")

print("- Extracting SS annotations")
ss_annots = []
for fn in category_fns:
    label = fn.split("/")[1].split("_")[0]
    with open(fn, "r") as f:
        annots = f.readlines()
        annots = [annot.strip('\n') + " {}".format(label) for annot in annots]
        ss_annots += annots
random.shuffle(ss_annots)

print("- Simplifying SS annotations")
ss_annots = [item.split() for item in ss_annots] # split 'video_validation_0000365 56.1 59.6' into ['video_validation_0000365', '56.1', '59.6']

# train/val split
print("- Performing train/test split of SS annotations")
train_ss_annots = ss_annots[0 : int(len(ss_annots) * .9)]
val_ss_annots = ss_annots[int(len(ss_annots) * .9) : ]

# extract actions from long videos into short clips and rename all clips
print("-- Extracting subclips for ss_train_csv")
ss_train_csv, id_counter = extract_subclips(train_ss_annots, id_counter, "ss")
print("-- Extracting subclips for ss_val_csv")
ss_val_csv, id_counter = extract_subclips(val_ss_annots, id_counter, "ss")

# save strongly supervised training and validation CSVs to disk
print("- Writing SS train and val CSVs")
with open("thumos-ss-train.csv", "w+") as f:
    f.write("\n".join(ss_train_csv))
with open("thumos-ss-validation.csv", "w+") as f:
    f.write("\n".join(ss_val_csv))

##########################################
# FEW SHOT
##########################################
print("===== FEW SHOT =====")

print("- Extracting FS annotations")
train_fs_annots = []
val_fs_annots = []
test_fs_annots = []
for fn in fs_category_fns:
    label = fn.split("/")[1].split("_")[0]
    with open(fn, "r") as f:
        annots = f.readlines()
        annots = [annot.strip('\n') + " {}".format(label) for annot in annots]
        random.shuffle(annots)
        train_fs_annots += annots[0:5] # randomly select 5 samples each from few-shot class
        val_fs_annots += annots[5:40] # maintain some validation samples
        test_fs_annots += annots[40:] # keep the rest for training

# shuffle for good measure
random.shuffle(train_fs_annots)
random.shuffle(val_fs_annots)
random.shuffle(test_fs_annots)

# split 'video_validation_0000365 56.1 59.6' into ['video_validation_0000365', '56.1', '59.6']
print("- Simplifying FS annotations")
train_fs_annots = [item.split() for item in train_fs_annots]
val_fs_annots = [item.split() for item in val_fs_annots]
test_fs_annots = [item.split() for item in test_fs_annots]

# extract actions from long videos into short clips and rename all clips
print("-- Extracting subclips for fs_train_csv")
fs_train_csv, id_counter = extract_subclips(train_fs_annots, id_counter, "fs")
print("-- Extracting subclips for fs_val_csv")
fs_val_csv, id_counter = extract_subclips(val_fs_annots, id_counter, "fs")
print("-- Extracting subclips for fs_test_csv")
fs_test_csv, id_counter = extract_subclips(test_fs_annots, id_counter, "fs")

# save few-shot training, validation, and test CSVs to disk
print("- Writing FS train, val, and test CSVs")
with open("thumos-fs-train.csv", "w+") as f:
    f.write("\n".join(fs_train_csv))
with open("thumos-fs-validation.csv", "w+") as f:
    f.write("\n".join(fs_val_csv))
with open("thumos-fs-test.csv", "w+") as f:
    f.write("\n".join(fs_test_csv))
