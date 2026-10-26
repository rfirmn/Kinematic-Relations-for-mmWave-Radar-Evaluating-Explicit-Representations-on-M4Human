"""Synthetic software checks only; these do not validate M4Human or CUDA."""
import pytest
from relational64.pilot_review import select_validation_panel, aggregate_physical, review_run


def window(subject,action,index=0,split='val'):
    return {'window_uid':f'{subject}/{action}/{index}','subject_id':str(subject),'split':split,
            'source_keys':[f'[{subject}, {action}, {index+t}]' for t in range(64)]}


def test_panel_is_order_independent_uses_sensor_ids_and_common_activities_only():
    rows=[window(s,a,i) for s in (1,2) for a in (3,4) for i in (0,80)]
    rows += [window(1,9),window(99,99,split='test'),window(98,98,split='train')]
    a=select_validation_panel(rows,{'1','2'},activities=2)
    b=select_validation_panel(list(reversed(rows)),{'1','2'},activities=2)
    assert a==b and len(a)==4
    assert {r['subject_id'] for r in a}=={'1','2'}
    assert all(r['split']=='val' and '/9/' not in r['window_uid'] for r in a)


def test_validation_selection_refuses_cross_subject_and_cross_activity_windows():
    bad=window(1,3);bad['source_keys'][0]='[99, 3, 0]'
    with pytest.raises(ValueError,match='boundary'):select_validation_panel([bad],{'1'},1)
    bad=window(1,3);bad['source_keys'][0]='[1, 4, 0]'
    with pytest.raises(ValueError,match='boundary'):select_validation_panel([bad],{'1'},1)
    with pytest.raises(ValueError,match='Unexpected'):select_validation_panel([window(99,3)],{'1'},1)


def test_panel_coverage_shortfall_is_not_silently_reduced():
    with pytest.raises(ValueError,match='Insufficient'):
        select_validation_panel([window(1,3),window(2,4)],{'1','2'},1)


def test_physical_aggregation_weights_valid_observations_and_keeps_nonfinite_failures():
    def metric(n,count,bad=0):return {'metrics':{'global_mpjpe':{'numerator':n,'count':count,'unit':'m','nonfinite_prediction_count':bad}}}
    good=aggregate_physical([metric(1.,1),metric(9.,3)])['global_mpjpe']
    assert good['mean']==2.5 and good['mean_mm']==2500 and good['count']==4
    bad=aggregate_physical([metric(1.,1),metric(None,2,1)])['global_mpjpe']
    assert bad['mean'] is None and bad['numerator'] is None and bad['count']==3


@pytest.mark.parametrize('kwargs',[{'run_id':'../other'},{'run_id':'ok','activities':21},{'run_id':'ok','probe_steps':0}])
def test_review_budget_and_paths_checked_before_any_data_or_device_access(kwargs):
    with pytest.raises(ValueError):review_run({},**kwargs)


def test_saved_diagnostic_review_never_becomes_scientific_acceptance_and_detects_stale_input(tmp_path,monkeypatch):
    from relational64 import config,reporting
    from relational64.config import write_json,identity,file_hash
    monkeypatch.setattr(config,'ROOT',tmp_path)
    artifact=tmp_path/'candidate.json';write_json(artifact,{'candidate':True})
    folder=tmp_path/'diagnostic_runs/run/reviews/review'
    plan={'artifact_generation_id':'current','dependencies':{str(artifact):file_hash(artifact)},
          'module_sha256':file_hash(reporting.Path(reporting.__file__).with_name('pilot_review.py'))}
    write_json(folder/'plan.json',plan)
    summary={'run_id':'run','review_id':'review','status':'passed','execution_complete':True,
             'diagnostic_only':True,'scientific_acceptance':False,'plan_hash':identity(plan),'validation_QA_count':12}
    write_json(folder/'summary.json',summary)
    rows=reporting.diagnostic_review_rows(tmp_path,'current')
    assert rows[0]['status']=='passed' and rows[0]['execution_complete']
    write_json(artifact,{'changed':True})
    rows=reporting.diagnostic_review_rows(tmp_path,'current')
    assert rows[0]['status']=='stale_dependencies' and not rows[0]['execution_complete']
    summary['scientific_acceptance']=True;write_json(folder/'summary.json',summary)
    with pytest.raises(ValueError,match='scientific acceptance'):
        reporting.diagnostic_review_rows(tmp_path,'current')
