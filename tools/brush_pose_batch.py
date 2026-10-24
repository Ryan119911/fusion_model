"""Finite batch sequencing; backends retain responsibility for physical guards."""


def run_pose_batch(jobs, backend):
    report={'status':'preflight','motion_sent':False,'completed_samples':[]}
    try:
        if not jobs or len(jobs)>27:
            raise ValueError('finite selected calibration batch required')
        for job in jobs:
            kinds=[s['kind'] for s in job['stages']]
            if kinds[0]!='transfer' or kinds[-1]!='write' or any(k!='air_rotation' for k in kinds[1:-1]):
                raise ValueError('each trial requires transfer, air references, fixed-orientation write')
            if len(kinds)<3:raise ValueError('actual local unloaded reference required before writing')
        backend.preflight_all(jobs)
        for job in jobs:
            report['active_sample']=job['sample_id']
            backend.event({'event':'sample_starting','sample_id':job['sample_id']})
            for index,stage in enumerate(job['stages']):
                if stage['kind']=='air_rotation':backend.air_reference(job,stage,index)
                else:backend.held_orientation(job,stage,index)
            report['completed_samples'].append(job['sample_id'])
            backend.event({'event':'sample_completed','sample_id':job['sample_id'],
                           'completed_count':len(report['completed_samples'])})
        report['status']='selected_pose_depth_samples_written_waiting_shared_capture'
    except BaseException as exc:
        report.update(status='stopped_on_error',error=type(exc).__name__+': '+str(exc))
    finally:
        report['motion_sent']=backend.motion_sent
    return report
