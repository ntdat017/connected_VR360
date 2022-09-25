
from pathlib import Path
import argparse
import cv2
import glob
import matplotlib.cm as cm
import torch
import py360convert

from connectedvr360.config import opt

from models.matching import Matching
from models.utils import (AverageTimer, make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

def load_model(opt=opt):
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    # print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    return Matching(config).eval().to(device)


def is_connected(img_path0, img_path1, model, opt=opt):
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    matching = model

    keys = ['keypoints', 'scores', 'descriptors']

    timer = opt.timer

    img0 = load_img(img_path0, opt.resize)
    img1 = load_img(img_path1, opt.resize)

    timer.update('data')

    frame_tensor = frame2tensor(img0, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor


    frame_tensor = frame2tensor(img1, device)
    pred = matching({**last_data, 'image1': frame_tensor})
    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()
    timer.update('forward')

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])

    n_matches = len(mkpts0)

    stem0, stem1 = img_path0.split('/')[-1][:-4], img_path1.split('/')[-1][:-4]

    ## visualize
    if opt.visualize:
        text = [
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']


        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {}:{}'.format(stem0, stem1),
        ]
        out = make_matching_plot_fast(
            img0, img1, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

        if opt.output_dir is not None:
            print('==> Will write outputs to {}'.format(opt.output_dir))
            Path(opt.output_dir).mkdir(exist_ok=True)

        if opt.output_dir is not None and n_matches > opt.n_matches_threshold:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = f'matches_{stem0}_{stem1}'
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

        timer.update('viz')
    timer.print()
    
    return (stem0, stem1, len(mkpts0))

    # if len(mkpts0) > opt.n_matches_threshold:
    #     return True
    # return False


def load_img(img_path, resize=None):
    img = cv2.imread(img_path, 0)
    # Equirectangular to Horizontal 
    
    if len(img.shape) == 2:
        img = img[..., None]
    cube = py360convert.e2c(img, face_w=256, mode='bilinear', cube_format='dice')
    h, w = cube.shape[:2]
    img = cube[h//3: h//3 *2, :, 0]

    # resize
    if resize and len(resize) == 2:
        img = cv2.resize(img, resize)
    return img