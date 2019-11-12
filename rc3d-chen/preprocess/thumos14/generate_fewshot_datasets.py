import shutil
from os.path import join, exists

import click

FEWSHOT_TRAIN_CLASSES = [
    'BaseballPitch',
    'CliffDiving',
    'HammerThrow',
    'VolleyballSpiking',
    'Billiards',
    'SoccerPenalty'
]

def load_classnames(thumos_path):
    """Loads list of all class names."""
    classnames = []
    with open(join(thumos_path, 'annotations_val/detclasslist.txt'), 'r') as f:
        for line in f.readlines():
            classnames.append(line.split(' ')[1].strip('\n'))
    return classnames

def copy_annotations_and_frames(dataset_name, src_paths, dest_paths, classnames):
    """ Copies annotations frames from src_paths to dest_paths.

    :param dataset_name: "train" or "test". Whether to create training set or test set.
    :param classnames: List of all classnames
    """
    for train_class in classnames:
        # Choose different output path depending on whether we are creating train or test set
        if dataset_name == 'train':
            output_annotation_path = join(dest_paths['val_annotations'], f'{train_class}_val.txt')
        else:
            output_annotation_path = join(dest_paths['test_annotations'], f'{train_class}_test.txt')

        output_annotation = open(output_annotation_path, 'w')
        print(f'\nCopying annotations and frames for {train_class}...\n')

        for suffix, annotations_dir, frames_dir in [
            ('val', src_paths['val_annotations'], src_paths['val_frames']),
            ('test', src_paths['test_annotations'], src_paths['test_frames'])]:
            annotation_filename = join(annotations_dir, f'{train_class}_{suffix}.txt')
            assert exists(annotation_filename)
            with open(annotation_filename, 'r') as input_annotation:
                for line in input_annotation.readlines():
                    output_annotation.write(line)
                    video_filename = line.split(' ')[0]
                    src_frames_path = join(frames_dir, video_filename)
                    if dataset_name == 'train':
                        dest_frames_path = join(dest_paths['val_frames'], video_filename)
                    else:
                        dest_frames_path = join(dest_paths['test_frames'], video_filename)
                    # Don't make duplicate copies
                    if exists(dest_frames_path):
                        continue

                    assert exists(src_frames_path)
                    print(f'Copying {src_frames_path} to {dest_frames_path}')
                    shutil.copytree(src_frames_path, dest_frames_path)

@click.command()
@click.option('--thumos-path', help='Path to THUMOS14 dataset', type=str, required=True)
@click.option('--output-path', help='Output path where fewshot training and test annotations and frames will be written to', type=str, required=True)
def main(thumos_path, output_path):
    all_classnames = load_classnames(thumos_path)

    src_paths = {}
    dest_paths = {}
    for var_name, path in [(src_paths, thumos_path), (dest_paths, output_path)]:
        var_name['frames'] = join(path, 'frames')
        var_name['val_annotations'] = join(path, 'annotations_val')
        var_name['val_frames'] = join(var_name['frames'], 'val')
        var_name['test_annotations'] = join(path, 'annotations_test')
        var_name['test_frames'] = join(var_name['frames'], 'test')

        # All paths must be valid directories
        for k in var_name:
            assert exists(var_name[k])

    copy_annotations_and_frames('train', src_paths, dest_paths, FEWSHOT_TRAIN_CLASSES)

    test_classnames = set(all_classnames) - set(FEWSHOT_TRAIN_CLASSES)
    print(f'List of test classes: {test_classnames}')
    copy_annotations_and_frames('test', src_paths, dest_paths, test_classnames)

if __name__ == '__main__':
    main()
