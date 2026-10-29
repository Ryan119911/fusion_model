"""Experiment-local current CAD geometry adapter; never changes formal YAML files.

Uses the existing Pinocchio/hppfcl attachment-to-arm checker. Keeps each source
model's force-sensor/gripper envelopes and adds the current CAD boxes. Only the
obsolete camera/bracket boxes are replaced by camera/retained-connector CAD.
"""
import copy
import numpy as np


def model_for_current_mount(config, boxes, base_class=None):
    import pinocchio as pin
    from robot_attachments import load
    if base_class is None:
        from self_collision import SelfCollisionModel as base_class
    geometry=load(config,root='/home/jx/UR10')
    model=base_class(geometry=geometry)
    frame=model.model.frames[model.model.getFrameId('tool0')]
    placement=frame.placement
    for old_name,new_name in [('camera_housing','mecheye_camera'),('camera_bracket','connector')]:
        index=next(i for name,i in model.attachments if name==old_name)
        data=boxes[new_name];obj=model.geometry.geometryObjects[index]
        obj.geometry=pin.hppfcl.Box(*data['size_m'])
        obj.placement=placement*pin.SE3(np.array(data['rotation']),np.array(data['centre_m']))
    added=[]
    for name in ['force_sensor','robotiq_2f85']:
        data=boxes[name]
        index=model.geometry.addGeometryObject(pin.GeometryObject('current_cad_'+name,frame.parentJoint,
            placement*pin.SE3(np.array(data['rotation']),np.array(data['centre_m'])),pin.hppfcl.Box(*data['size_m'])))
        added.append(('current_cad_'+name,index))
        for arm in model.robot_ids:model.geometry.addCollisionPair(pin.CollisionPair(index,arm))
    model.attachments+=added
    model.camera_ids=tuple(i for _,i in model.attachments)
    model.camera_names=tuple(n for n,_ in model.attachments)
    model.geometry_data=model.geometry.createData()
    return model
