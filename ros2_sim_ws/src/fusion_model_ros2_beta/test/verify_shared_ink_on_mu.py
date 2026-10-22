"""Explicit no-motion integration check; run on an isolated ROS_DOMAIN_ID."""
import csv
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import rclpy
from fusion_model_ros2_beta.brush_trajectory_driver import BrushTrajectoryDriver
from fusion_model_ros2_beta.trajectory_catalog import TrajectoryEntry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', default='vertical', choices=['vertical', 'horizontal'])
    args = parser.parse_args()
    rclpy.init(args=['--ros-args', '-r', '__node:=shared_ink_verification',
                      '-p', 'enable_input_page:=false'])
    node = BrushTrajectoryDriver()
    try:
        root = Path('/home/robot/coppeliasim/machine_learning/model/outputs/ros2_v16_fontsize_cache')
        entries = []
        for char, cache in [('武', 'char_u6b66/font_0120mm_0133057e7973'),
                            ('汉', 'char_u6c49/font_0120mm_4b0b259475d6')]:
            folder = root/cache
            node._offline_generator._ensure_neural_ink(folder)
            with (folder/'physical_trajectory.csv').open() as f:
                sample = next(csv.DictReader(f))['sample_id']
            source = TrajectoryEntry(character=char, status='ready', sample_id=sample)
            entries.append(node._offline_generator._entry(source, folder/'physical_trajectory.csv',
                            folder/'scaled_target.png', folder, .12))
        # Does not spin timers or call _publish_once: never sends joint commands.
        node._build_targets(entries=entries, layout_mode=args.layout, font_size_m=.12)
        node._publish_ink(-1)
        assert not node._ink_markers.markers
        second_start = node._neural_jobs[1]['schedule'][0]
        node._publish_ink(int(second_start)-1)
        assert len(node._ink_markers.markers) == 1
        node._publish_ink(len(node._targets)-1)
        assert len(node._ink_markers.markers) == 2
        for job in node._neural_jobs:
            frame = job['frames'][-1]
            recovered = np.zeros_like(frame)
            marker = job['marker']
            for i in range(0, len(marker.points), 6):
                quad = marker.points[i:i+6]
                x = sum(p.x for p in quad)/6
                y = sum(p.y for p in quad)/6
                col = round((x-job['origin'][0])/job['pixel_size'])
                row = round((job['origin'][1]-y)/job['pixel_size'])
                recovered[row,col] = marker.colors[i].a
            np.testing.assert_array_equal(recovered, frame)
        result = dict(targets=len(node._targets), glyphs=[dict(character=j['character'],
                      frames=len(j['frames']), audit=j['audit']) for j in node._neural_jobs],
                      marker_roundtrip_exact=True, second_glyph_not_shown_early=True)
        node._clear_ink()
        assert not node._neural_jobs and not node._ink_markers.markers
        result['clear_verified'] = True
        print(json.dumps(result, ensure_ascii=False))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
