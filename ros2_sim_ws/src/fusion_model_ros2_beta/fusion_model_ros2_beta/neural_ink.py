"""ROS-independent geometry and ordered playback for the shared neural ink."""
import numpy as np


def frame_quads(frame, origin, pixel_size):
    # Do not threshold or rescale opacity: preserve every nonzero model pixel.
    ys, xs = np.nonzero(frame > 0)
    x0 = origin[0] + (xs - 0.5)*pixel_size
    y0 = origin[1] - (ys - 0.5)*pixel_size
    corners = np.stack((np.stack((x0,y0),axis=-1),
                        np.stack((x0+pixel_size,y0),axis=-1),
                        np.stack((x0+pixel_size,y0-pixel_size),axis=-1),
                        np.stack((x0,y0-pixel_size),axis=-1)),axis=1)
    return corners[:, [0,2,1,0,3,2]], frame[ys,xs]


def schedule_frames(sample_xy, sample_ids, targets, offset):
    """Map ordered model samples onto the same stroke's commanded path.

    Nearest locations are constrained to progress monotonically; airborne
    targets never deposit ink and separate characters have separate IDs.
    """
    indices = []
    floor = 0
    for xy, stroke in zip(sample_xy, sample_ids):
        candidates = [(i,t.point) for i,t in enumerate(targets)
                      if i>=floor and t.point.state!=3 and t.point.stroke_id==int(stroke)+offset]
        if not candidates:
            raise ValueError('Ink stream cannot be associated with commanded stroke')
        index, _ = min(candidates, key=lambda item: (item[1].x-xy[0])**2+(item[1].y-xy[1])**2)
        floor = index
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)
