from os import listdir
from os.path import join

import click

from generate_fewshot_datasets import FEWSHOT_TRAIN_CLASSES

FRAMERATE = 25.0

def load_classnames(classnames_path):
    classnames = []
    with open(classnames_path, 'r') as f:
        for line in f.readlines():
            classnames.append(line.split(' ')[1].strip('\n'))
    return classnames

@click.command()
@click.option('--ucf-frames-path', required=True)
@click.option('--class-list-path', required=True)
@click.option('--dataset-output-path', required=True)
def main(ucf_frames_path, class_list_path, dataset_output_path):
    class_list = load_classnames(class_list_path)
    annotations_output_path = join(dataset_output_path, 'annotations_val')

    # List of names of all directories containing frames, such as v_SoccerPenalty_g09_c04
    frames_dir_names = listdir(ucf_frames_path)

    for classname in class_list:
        if classname not in FEWSHOT_TRAIN_CLASSES:
            filtered_dir_names = [name for name in frames_dir_names if classname in name]
            filtered_dir_names.sort()
            with open(join(annotations_output_path, f'{classname}_val.txt'), 'w') as class_annotation_file:
                for video_name in filtered_dir_names:
                    # Count number of frames
                    num_frames = len(listdir(join(ucf_frames_path, video_name)))
                    end_timestamp = int(num_frames / FRAMERATE * 100) / 100.0
                    class_annotation_file.write(f'{video_name} 0.0 {end_timestamp}\n')

if __name__ == '__main__':
    main()
