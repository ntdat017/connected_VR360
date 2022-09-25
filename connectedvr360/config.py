
opt = {
    'output_dir': 'dump',
    'image_glob': ['*.jpg'],
    'skip': 1,
    'max_length': 10000,
    'resize': None,
    'superglue': 'indoor',
    'max_keypoints': -1,
    'keypoint_threshold': 0.1,
    'nms_radius': 4,
    'sinkhorn_iterations': 20,
    'match_threshold': 0.2,
    'show_keypoints': True,
    'force_cpu': True,
    'n_matches_threshold': 20, 
    'visualize': False
}


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

opt = Struct(**opt)