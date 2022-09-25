from sys import argv
import argparse

from pathlib import Path
import glob
import os
import json

from connectedvr360.utils import is_connected, load_model
from models.utils import AverageTimer


def main():
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--image_dir', type=str, default='0',
        help='input image path')
    parser.add_argument(
        '--output_dir', type=str, default='dump_vis',
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=(640, 480),
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--n_matches_threshold', type=int, default=20,
        help='number match threshold')

    parser.add_argument(
        '--visualize', action='store_true', default=False,
        help='is visualize')

    args = parser.parse_args()
    
    if not Path(args.image_dir).is_dir():
        raise IOError('No image dir found')

    img_paths = []
    for ext in args.image_glob:
        img_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))

    if len(img_paths) == 0:
        raise IOError('No images found (maybe bad \'image_glob\' ?)')

    matching = load_model(opt=args)

    timer = AverageTimer()

    d = vars(args)
    d['timer'] = timer

    results = []
    for i in range(len(img_paths)):
        for j in range(i, len(img_paths)):
            img_path0 = img_paths[i]
            img_path1 = img_paths[j]
            result = is_connected(img_path0, img_path1, matching,  args)
            if result[-1] > args.n_matches_threshold:
                results.append(result)
        break

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()