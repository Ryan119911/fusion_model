"""Offline ink/side-view measurements and explicit-input response calibration."""
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from brush_longline_calibration import digest, read, save, write_csv


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def feedback_samples(path, plan):
    g = plan['config']['geometry']
    axes = np.column_stack([g[k] for k in ('paper_u_base', 'paper_v_base', 'paper_normal_base')])
    origin, tip = np.array(g['paper_origin_base_m']), np.array(g['tip_tool_m'])
    R0 = Rotation.from_rotvec(g['baseline_rotation_rad']).as_matrix()
    rows = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line); pose = np.array(row['actual_tool0_pose_m_rad'], float)
        if pose.shape != (6,) or not np.isfinite(pose).all(): raise ValueError('invalid actual feedback pose')
        rot = Rotation.from_rotvec(pose[3:]).as_matrix()
        local = axes.T @ (pose[:3] + rot @ tip - origin) * 1000
        angles = Rotation.from_matrix(axes.T @ rot @ R0.T @ axes).as_euler('xyz')
        velocity = np.array(row.get('tcp_velocity', [0.] * 6))
        tip_velocity = velocity[:3] + np.cross(velocity[3:], rot @ tip)
        rows.append({'host_monotonic_s': float(row['host_monotonic_s']), 's_mm': float(local[0] - plan['config']['motion']['start_uv_mm'][0]),
                     'H_exec_mm': float(-local[2]), 'alpha_exec_rad': float(angles[0]), 'beta_exec_rad': float(angles[1]),
                     'residual_angle_rad': float(angles[2]), 'forward_speed_mm_s': float(tip_velocity @ axes[:, 0] * 1000)})
    t = np.array([r['host_monotonic_s'] for r in rows])
    if len(rows) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('feedback requires finite strictly increasing host_monotonic_s')
    return rows


def ink_profile(before, after, calibration, plan, feedback=None):
    before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY) if before.ndim == 3 else before
    after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY) if after.ndim == 3 else after
    if before.shape != after.shape: raise ValueError('before/after image dimensions differ')
    matrix = np.asarray(calibration.get('image_to_paper_uv_mm',calibration.get('after_image_to_paper_uv_mm')), float)
    before_matrix = np.asarray(calibration.get('before_image_to_paper_uv_mm',matrix),float)
    bounds = np.array(calibration['roi_uv_mm'], float)
    scale = float(calibration['mm_per_pixel'])
    if any(m.shape != (3,3) or not np.isfinite(m).all() or abs(np.linalg.det(m))<1e-12 for m in (matrix,before_matrix)) or bounds.shape != (4,) or not np.isfinite(bounds).all() or not scale > 0 or not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
        raise ValueError('invalid metric image calibration')
    size = (int(math.ceil((bounds[1]-bounds[0])/scale)), int(math.ceil((bounds[3]-bounds[2])/scale)))
    if min(size) < 2 or max(size) > 20000: raise ValueError('invalid rectified image size')
    grid = np.array([[1/scale, 0, -bounds[0]/scale], [0, -1/scale, bounds[3]/scale], [0, 0, 1.]])
    transform = grid @ matrix
    rect_before = cv2.warpPerspective(before, grid @ before_matrix, size, flags=cv2.INTER_LINEAR, borderValue=255)
    rect_after = cv2.warpPerspective(after, transform, size, flags=cv2.INTER_LINEAR, borderValue=255)
    valid = cv2.warpPerspective(np.full(before.shape,255,np.uint8), transform, size, flags=cv2.INTER_NEAREST, borderValue=0) > 0
    valid &= cv2.warpPerspective(np.full(before.shape,255,np.uint8), grid @ before_matrix, size, flags=cv2.INTER_NEAREST, borderValue=0) > 0
    threshold = float(calibration.get('difference_threshold', 25))
    if not 0 < threshold < 255: raise ValueError('difference threshold must be 0..255')
    mask = ((rect_before.astype(float)-rect_after.astype(float) > threshold) & valid).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    minimum_area = float(calibration.get('minimum_component_area_mm2', .5)) / scale**2
    components = [i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA] >= minimum_area]
    if len(components) != 1: raise ValueError('expected one isolated ink component; found ' + str(len(components)))
    mask = (labels == components[0]).astype(np.uint8)
    if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
        raise ValueError('ink touches ROI edge; enlarge or correct the ROI')
    x0, y0 = plan['config']['motion']['start_uv_mm']
    rows = []
    actual = None
    skipped = 0
    if feedback is not None:
        candidates = [r for r in feedback if r['forward_speed_mm_s'] > .05 * plan['config']['motion']['speed_mm_s'] and 0 <= r['s_mm'] <= plan['config']['motion']['length_mm']]
        actual = []
        for r in candidates:
            if actual and r['s_mm'] < actual[-1]['s_mm'] - float(calibration.get('maximum_projection_reversal_mm', .2)):
                raise ValueError('actual scan projection reversed; inspect feedback')
            if actual and r['s_mm'] - actual[-1]['s_mm'] < float(calibration.get('minimum_forward_increment_mm', .05)):
                skipped += 1; continue
            actual.append(r)
        if len(actual) < 2: raise ValueError('insufficient advancing actual scan samples')
    alignment = calibration.get('input_alignment', {})
    verified = alignment.get('verified') is True
    if verified and (not alignment.get('evidence') or actual is None):
        raise ValueError('verified input alignment needs actual feedback and an evidence description')
    bin_pixels = max(1, int(round(float(calibration.get('bin_mm', .5))/scale)))
    for left in range(0,size[0],bin_pixels):
        widths, centers = [], []
        for x in range(left,min(left+bin_pixels,size[0])):
            ys = np.flatnonzero(mask[:,x])
            if len(ys):
                if np.any(np.diff(ys)>1): continue
                widths.append((ys[-1]-ys[0]+1)*scale)
                centers.append(bounds[3]-(ys[0]+ys[-1]+1)*scale/2-y0)
        if not widths: continue
        u = bounds[0]+(left+min(left+bin_pixels,size[0]))/2*scale
        position = u-x0
        sample = {'trial_id':plan['trial_id'],'split':plan['split'],'repeat_group':plan['repeat_group'],
                  's_mm':float(position),'H_exec_mm':'','alpha_exec_rad':'','beta_exec_rad':'',
                  'width_mm':float(np.median(widths)),'center_offset_mm':float(np.median(centers)),
                  'status':'valid','input_correspondence':'verified' if verified else 'unverified', 'data_kind':'measured',
                  'feedback_nonadvancing_samples_skipped':skipped}
        if actual is not None:
            q = position+float(alignment.get('ink_to_nominal_tip_offset_mm',0))
            locations = np.array([r['s_mm'] for r in actual])
            j = int(np.searchsorted(locations,q))
            if j == 0 or j == len(locations): sample['status']='outside_feedback_span'
            else:
                gap = actual[j]['host_monotonic_s']-actual[j-1]['host_monotonic_s']
                if gap > float(calibration.get('maximum_feedback_gap_s', .15)): sample['status']='feedback_gap'
                else:
                    for key in ('H_exec_mm','alpha_exec_rad','beta_exec_rad'):
                        sample[key]=float(np.interp(q,locations,[r[key] for r in actual]))
        rows.append(sample)
    if not rows: raise ValueError('no valid width cross sections')
    return rows, rect_after, mask*255


def side_features(video, frame_log, calibration):
    rows = [json.loads(line) for line in Path(frame_log).read_text().splitlines()]
    cap = cv2.VideoCapture(str(video)); results=[]
    roi = np.asarray(calibration['brush_roi_px'],int)
    normal = np.asarray(calibration['paper_line_abc'],float)
    if roi.shape != (4,) or normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal[:2]) <= 0:
        raise ValueError('side ROI and oriented paper line ax+by+c required')
    normal /= np.linalg.norm(normal[:2])
    scale = float(calibration['mm_per_pixel'])
    threshold = float(calibration.get('dark_threshold',100))
    if not scale>0 or not 0<threshold<255: raise ValueError('invalid side scale/threshold')
    try:
        for index, record in enumerate(rows):
            ok,image=cap.read()
            if not ok: raise ValueError('side video has fewer frames than timestamp log')
            if record['video_frame_index'] != index: raise ValueError('video frame indices are not consecutive')
            x1,y1,x2,y2=roi
            if not 0<=x1<x2<=image.shape[1] or not 0<=y1<y2<=image.shape[0]: raise ValueError('side ROI outside frame')
            gray=cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
            foreground=(gray<threshold).astype(np.uint8)
            count,labels,stats,_=cv2.connectedComponentsWithStats(foreground,8)
            candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=calibration.get('minimum_brush_area_px',20)]
            result={'video_frame_index':index,'host_monotonic_s':record['host_monotonic_s'],'tip_gap_mm':'','tip_x_px':'','tip_y_px':'','status':'not_visible'}
            if candidates:
                selected=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
                yy,xx=np.where(labels==selected);xx=xx+x1;yy=yy+y1
                distances=normal[0]*xx+normal[1]*yy+normal[2]
                # Positive signed distance must be above the paper, by calibration.
                point=int(np.argmin(distances))
                result.update(tip_gap_mm=float(distances[point]*scale),tip_x_px=int(xx[point]),tip_y_px=int(yy[point]),status='visible')
            results.append(result)
        if cap.read()[0]: raise ValueError('side video has more frames than timestamp log')
    finally:
        cap.release()
    return results


def estimate_zero(side_rows, feedback, plan, threshold=.1, consecutive=3):
    if not math.isfinite(threshold) or threshold<0 or consecutive<2: raise ValueError('invalid contact event settings')
    t=np.array([r['host_monotonic_s'] for r in feedback]); depth=np.array([r['H_exec_mm'] for r in feedback])
    if len(t)<2 or np.any(np.diff(t)<=0): raise ValueError('feedback time must increase')
    run=[];last_clear=None
    for row in side_rows:
        if row['status']!='visible': run=[];continue
        stamp=float(row['host_monotonic_s']);gap=float(row['tip_gap_mm'])
        if gap>threshold:
            last_clear=stamp;run=[];continue
        run.append(stamp)
        if len(run)<consecutive: continue
        if last_clear is None: raise ValueError('no clearly separated frame before contact; zero is unidentified')
        contact=run[0]
        if not t[0]<=last_clear<contact<=t[-1]: raise ValueError('contact event outside feedback time span')
        # Do not bridge a missing state stream when estimating the zero.
        first=max(0,int(np.searchsorted(t,last_clear))-1);last=min(len(t)-1,int(np.searchsorted(t,contact)))
        if np.any(np.diff(t[first:last+1])>.15): raise ValueError('feedback gap around contact')
        H=float(np.interp(contact,t,depth)); H_before=float(np.interp(last_clear,t,depth))
        normal=np.array(plan['config']['geometry']['paper_normal_base']);origin=np.array(plan['config']['geometry']['paper_origin_base_m'])
        return {'schema':'brush_longline_visual_zero_v1','trial_id':plan['trial_id'],
                'zero_kind':'visual_contact_candidate','source_config_sha256':plan['config_sha256'],
                'last_clear_monotonic_s':last_clear,'first_contact_monotonic_s':contact,
                'H_at_contact_mm':H,'H_last_clear_mm':H_before,'paper_normal_offset_mm':-H,
                'candidate_paper_origin_base_m':(origin-normal*H/1000).tolist(),
                'scope':'projected threshold contact candidate, not independently established zero-deformation truth',
                'motion_sent':False}
    raise ValueError('no sustained visible contact event')


def basis(x):
    h,a,b=np.asarray(x).T
    return np.column_stack((np.ones(len(h)),h,a,b,h*h,a*a,b*b,h*a,h*b,a*b))


def handeye_images(images, calibration, plan, before_pose, after_pose):
    if before_pose is None or after_pose is None: raise ValueError('handeye mode requires both measured photo poses')
    info=calibration['camera_info'];K=np.array(info['k'],float).reshape(3,3);D=np.array(info['d'],float)
    if info['distortion_model'] not in ('','plumb_bob','rational_polynomial') or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0:
        raise ValueError('unsupported/invalid camera calibration')
    if info.get('binning_x',0)>1 or info.get('binning_y',0)>1 or info.get('roi',{}).get('x_offset',0) or info.get('roi',{}).get('y_offset',0):
        raise ValueError('binned/cropped CameraInfo requires adjusted intrinsics')
    tool_camera=np.array(calibration['tool0_from_camera'],float)
    if tool_camera.shape!=(4,4) or not np.isfinite(tool_camera).all() or not np.allclose(tool_camera[3],[0,0,0,1]) or not np.allclose(tool_camera[:3,:3].T@tool_camera[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(tool_camera[:3,:3])<.999:
        raise ValueError('explicit finite tool0_from_camera matrix required')
    g=plan['config']['geometry'];u,v,n=[np.array(g[k]) for k in ('paper_u_base','paper_v_base','paper_normal_base')]
    origin=np.array(g['paper_origin_base_m']);output=[];new=dict(calibration)
    new.pop('image_to_paper_uv_mm',None)
    for image,path,key in zip(images,(before_pose,after_pose),('before_image_to_paper_uv_mm','after_image_to_paper_uv_mm')):
        if image.shape[:2]!=(info['height'],info['width']): raise ValueError('image/CameraInfo dimensions differ')
        pose=np.array(read(path)['flange_pose_m_rad'],float)
        if pose.shape!=(6,) or not np.isfinite(pose).all(): raise ValueError('finite actual photo pose required')
        tool=np.eye(4);tool[:3,:3]=Rotation.from_rotvec(pose[3:]).as_matrix();tool[:3,3]=pose[:3]
        camera=tool@tool_camera;R=camera[:3,:3].T
        bounds=calibration['roi_uv_mm']
        corners=np.array([origin+x/1000*u+y/1000*v for x in bounds[:2] for y in bounds[2:]])
        if np.any(((corners-camera[:3,3])@camera[:3,:3])[:,2]<=0): raise ValueError('paper ROI is behind camera')
        paper_to_image=K@np.column_stack((R@u/1000,R@v/1000,R@(origin-camera[:3,3])))
        new[key]=np.linalg.inv(paper_to_image).tolist()
        output.append(cv2.undistort(image,K,D))
    return output,new


def fit_response(rows,ridge=1e-6):
    if not np.isfinite(ridge) or ridge<0: raise ValueError('ridge must be finite and nonnegative')
    rows=[r for r in rows if r.get('status')=='valid' and r.get('input_correspondence')=='verified']
    if not rows or any(r.get('split')!='train' for r in rows): raise ValueError('fit accepts verified train rows only; never include holdout observations')
    X=np.array([[r[k] for k in ('H_exec_mm','alpha_exec_rad','beta_exec_rad')] for r in rows],float)
    Y=np.array([[r[k] for k in ('width_mm','center_offset_mm')] for r in rows],float)
    if len(X)<10 or not np.isfinite(np.r_[X.ravel(),Y.ravel()]).all(): raise ValueError('insufficient finite data')
    center=(X.max(0)+X.min(0))/2;scale=(X.max(0)-X.min(0))/2
    if np.any(scale<1e-10): raise ValueError('all three inputs require independent excitation')
    normalized=(X-center)/scale;Phi=basis(normalized)
    if np.linalg.matrix_rank(Phi)<10: raise ValueError('quadratic response is not identifiable from this input design')
    coefficients=np.linalg.lstsq(np.vstack((Phi,np.sqrt(ridge)*np.eye(10))),np.vstack((Y,np.zeros((10,2)))),rcond=None)[0]
    hull=ConvexHull(normalized)
    error=Phi@coefficients-Y
    return {'schema':'brush_longline_response_v1','input_columns':['H_exec_mm','alpha_exec_rad','beta_exec_rad'],
            'output_columns':['width_mm','center_offset_mm'],'center':center.tolist(),'scale':scale.tolist(),
            'coefficients':coefficients.tolist(),'support_hull_equations':hull.equations.tolist(),
            'training_trials':sorted(set(r['trial_id'] for r in rows)),
            'training_repeat_groups':sorted(set(r['repeat_group'] for r in rows)),
            'data_kinds':sorted(set(r['data_kind'] for r in rows)),
            'training_rows':len(rows),'training_rmse_mm':np.sqrt(np.mean(error**2,axis=0)).tolist(),
            'scope':'fixed direction/speed static ink response; training residual is not independent validation'}


def predict_rows(model,rows):
    if model.get('schema')!='brush_longline_response_v1': raise ValueError('unsupported response model')
    result=[];errors=[]
    for row in rows:
        entry=dict(row)
        if row.get('status')!='valid' or row.get('input_correspondence')!='verified':
            entry.update(predicted_width_mm='',predicted_center_offset_mm='',width_error_mm='',center_error_mm='',in_training_hull='unavailable')
        else:
            x=np.array([row[k] for k in model['input_columns']],float)
            if not np.isfinite(x).all(): raise ValueError('nonfinite prediction input')
            z=(x-np.array(model['center']))/np.array(model['scale'])
            prediction=(basis(z[None,:])@np.array(model['coefficients']))[0]
            hull=np.array(model['support_hull_equations']);inside=bool(np.all(hull[:,:3]@z+hull[:,3]<=1e-8))
            entry.update(predicted_width_mm=float(prediction[0]),predicted_center_offset_mm=float(prediction[1]),
                         width_error_mm='',center_error_mm='',in_training_hull=inside)
            if row.get('width_mm','')!='' and row.get('center_offset_mm','')!='':
                error=prediction-np.array([row['width_mm'],row['center_offset_mm']],float)
                entry.update(width_error_mm=float(error[0]),center_error_mm=float(error[1]))
                if inside:errors.append(error)
        result.append(entry)
    metrics={'within_hull_evaluated_rows':len(errors),'same_training_trial_present':any(r.get('trial_id') in model['training_trials'] for r in rows),
             'same_training_repeat_group_present':any(r.get('repeat_group') in model['training_repeat_groups'] for r in rows)}
    if errors:
        error=np.array(errors)
        metrics.update(mae_mm=np.abs(error).mean(0).tolist(),rmse_mm=np.sqrt((error**2).mean(0)).tolist(),absolute_p95_mm=np.quantile(np.abs(error),.95,axis=0).tolist())
    return result,metrics


def dispatch(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    if args.command=='measure':
        plan=read(args.plan);calibration=read(args.calibration)
        images=[cv2.imread(str(p)) for p in (args.before,args.after)]
        if any(x is None for x in images): raise ValueError('cannot read before/after image')
        if calibration.get('mode')=='handeye':
            images,calibration=handeye_images(images,calibration,plan,args.before_pose,args.after_pose)
        feedback=feedback_samples(args.feedback,plan) if args.feedback else None
        rows,rectified,mask=ink_profile(*images,calibration,plan,feedback)
        write_csv(out/'features.csv',rows);cv2.imwrite(str(out/'rectified_after.png'),rectified);cv2.imwrite(str(out/'ink_mask.png'),mask)
        result={'status':'ink_measured','rows':len(rows),'verified_input_rows':sum(r['input_correspondence']=='verified' and r['status']=='valid' for r in rows),'motion_sent':False}
    elif args.command=='side':
        rows=side_features(args.video,args.frames,read(args.calibration));write_csv(out/'side_features.csv',rows)
        result={'status':'side_projection_measured','frames':len(rows),'motion_sent':False}
    elif args.command=='zero':
        plan=read(args.plan)
        result=estimate_zero(csv_rows(args.side_csv),feedback_samples(args.feedback,plan),plan,args.gap_threshold_mm,args.consecutive_frames)
        save(out/'zero.json',result)
    elif args.command=='fit':
        rows=[row for file in args.csv for row in csv_rows(file)]
        model=fit_response(rows,args.ridge);save(out/'model.json',model)
        result={'status':'static_response_fitted','training_rows':model['training_rows'],'data_kinds':model['data_kinds'],'motion_sent':False}
    elif args.command=='predict':
        predicted,metrics=predict_rows(read(args.model),csv_rows(args.csv));write_csv(out/'predictions.csv',predicted);save(out/'metrics.json',metrics)
        result={'status':'predicted','metrics':metrics,'motion_sent':False}
    elif args.command=='demo':
        rng=np.random.default_rng(6126);rows=[]
        for group in range(8):
            for _ in range(50):
                h,a,b=rng.uniform(.2,.8),rng.uniform(-.03,.03),rng.uniform(-.03,.03)
                rows.append({'trial_id':'synthetic_'+str(group),'repeat_group':'synthetic_'+str(group),'split':'train' if group<6 else 'holdout',
                             'H_exec_mm':h,'alpha_exec_rad':a,'beta_exec_rad':b,'width_mm':1+2*h+4*a+3*b+2*h*a,
                             'center_offset_mm':3*a-2*b,'status':'valid','input_correspondence':'verified','data_kind':'synthetic_demo'})
        train=[r for r in rows if r['split']=='train'];holdout=[r for r in rows if r['split']=='holdout']
        write_csv(out/'synthetic_train.csv',train);write_csv(out/'synthetic_holdout.csv',holdout)
        model=fit_response(train);save(out/'model.json',model)
        predictions,metrics=predict_rows(model,holdout);write_csv(out/'predictions.csv',predictions);save(out/'metrics.json',metrics)
        result={'status':'synthetic_demo_completed','not_real_brush_calibration':True,'metrics':metrics,'motion_sent':False,'camera_triggered':False}
    else:
        raise ValueError('unsupported processing command')
    save(out/'RESULT.json',result)
    return result
