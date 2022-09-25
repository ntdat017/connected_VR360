## Connected VR 360


### Setup

```python setup.py install```

command line
```
usage: connected_vr360 [-h] [--image_dir IMAGE_DIR] [--output_dir OUTPUT_DIR] [--image_glob IMAGE_GLOB [IMAGE_GLOB ...]]
                       [--skip SKIP] [--max_length MAX_LENGTH] [--resize RESIZE [RESIZE ...]] [--superglue {indoor,outdoor}]
                       [--max_keypoints MAX_KEYPOINTS] [--keypoint_threshold KEYPOINT_THRESHOLD] [--nms_radius NMS_RADIUS]
                       [--sinkhorn_iterations SINKHORN_ITERATIONS] [--match_threshold MATCH_THRESHOLD] [--show_keypoints]
                       [--no_display] [--force_cpu] [--n_matches_threshold N_MATCHES_THRESHOLD] [--visualize]
```

sample cmd:
```connected_vr360 --image_dir path-to-data/ --resize 480 240 --visualize```

result to `results.json` file