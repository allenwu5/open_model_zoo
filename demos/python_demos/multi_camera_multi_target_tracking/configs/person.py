random_seed = 100

obj_det = dict(
    trg_classes=(1,)
)

obj_segm = dict(
    trg_classes=(1,)
)

mct_config = dict(
    time_window=4,
    global_match_thresh=0.2,
    bbox_min_aspect_ratio=1.2
)

sct_config = dict(
    time_window=2,
    continue_time_thresh=1,
    track_clear_thresh=100,
    match_threshold=0.1,
    merge_thresh=0.15,
    n_clusters=8,
    max_bbox_velocity=1,
    detection_occlusion_thresh=0.7,
    track_detection_iou_thresh=0.1, 
    process_curr_features_number=0,
    interpolate_time_thresh=10,
    detection_filter_speed=0.6,
    rectify_thresh=0.1
)

normalizer_config = dict(
    enabled=False,
    clip_limit=.5,
    tile_size=8
)

visualization_config = dict(
    show_all_detections=True,
    max_window_size=(1920, 1080),
    stack_frames='vertical'
)

analyzer = dict(
    enable=False,
    show_distances=False,
    save_distances='',
    concatenate_imgs_with_distances=True,
    plot_timeline_freq=0,
    save_timeline='',
    crop_size=(32, 64)
)

embeddings = dict(
    save_path='',
    use_images=True,  # Use it with `analyzer['enable'] = True` to save crops of objects
    step=0  # Equal to subdirectory for `save_path`
)
