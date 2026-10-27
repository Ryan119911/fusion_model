"""Render offline CAD geometry only; no device access."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from camera_direct_mount import read_stl
ROOT=Path(__file__).resolve().parents[2]
src=Path(__file__).resolve().parent/'source_cad'
meshes=ROOT/'ros2_sim_ws/src/fusion_model_ros2_beta/meshes/end_effector'
out=ROOT/'outputs/camera_direct_mount_20261027'
out.mkdir(parents=True,exist_ok=True)
report=json.loads((Path(__file__).resolve().parent/'assembly.json').read_text())
cx,cy,cz=report['housing_centre_cad_mm']
fig=plt.figure(figsize=(12,6))
for k in range(2):
    ax=fig.add_subplot(1,2,k+1,projection='3d')
    for token,col in [('连接1','#a2abb4'),('六轴力','#205eb8'),('2F85','#27313a'),('3D','#d85820' if k==0 else '#467c98')]:
        if token=='3D':
            p=meshes/'end_effector_camera.stl' if k else next(src.glob('*3D相机*STL'))
        else:
            p=meshes/{'连接1':'end_effector_connector.stl','六轴力':'end_effector_force_sensor.stl','2F85':'end_effector_2f85_closed.stl'}[token]
        ax.add_collection3d(Poly3DCollection(read_stl(p)[:,:,[0,2,1]],facecolor=col,edgecolor='none',rasterized=True))
    ax.set(xlim=(0,420),ylim=(-40,220),zlim=(0,260),xlabel='CAD X (mm)',ylabel='CAD Z (mm)',zlabel='CAD Y (mm)')
    ax.set_box_aspect((420,260,260));ax.view_init(18,-65);ax.tick_params(labelsize=8)
    ax.set_title('Original assembly' if not k else 'Photo-based: long edge HORIZONTAL; camera-end tops flush',fontsize=11)
    if k: ax.quiver(cx,cz,cy-report['housing_size_cad_xyz_mm'][1]/2,0,0,-35,color='#15b4d4',arrow_length_ratio=.22,linewidth=2)
fig.subplots_adjust(left=.01,right=.97,bottom=.08,top=.88,wspace=.03)
fig.savefig(out/'assembly_comparison.png',dpi=120,bbox_inches='tight',pad_inches=.3)
print('preview ready')
