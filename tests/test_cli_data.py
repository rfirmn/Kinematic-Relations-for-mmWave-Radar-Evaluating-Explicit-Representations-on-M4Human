"""CLI routing and rejection fixtures; no real-data training or label reads."""
from pathlib import Path

import pytest

from relational64 import cli
from relational64.data import write_json


@pytest.mark.parametrize('command,extra',[
    ('audit',['--full']),('prepare',['--diagnostic']),('export-targets',['--split','train,val']),
    ('build-qa',['--split','train,val']),('verify',['--suite','data']),
    ('overfit',['--synthetic','--max-steps','1']),('profile',['--stage','physical','--synthetic']),
])
def test_data_commands_parser_smoke(command,extra):
    args=cli.parser().parse_args([command,*extra])
    assert args.command==command
    if command in ('export-targets','build-qa'):
        assert args.split=='train,val'


@pytest.fixture
def blocked_runtime(tmp_path,monkeypatch):
    from relational64 import config as config_module
    monkeypatch.setattr(config_module,'ROOT',tmp_path)
    inventory=tmp_path/'inventory.json'
    write_json(inventory,{'artifact_generation_id':'current','artifact_policy':'fresh_from_source_only','full_scan':True,'authority':{'G0_passed':False}})
    config={'runtime':{'artifact_generation_id':'current','artifact_policy':'fresh_from_source_only','derived_root':str(tmp_path/'derived'),'report_root':str(tmp_path/'reports'),'device':'cpu'}}
    monkeypatch.setattr(cli,'resolve',lambda path:config)
    return inventory,tmp_path


@pytest.mark.parametrize('command',['prepare','export-targets'])
def test_unapproved_real_data_commands_reject_before_output(command,blocked_runtime,capsys):
    inventory,root=blocked_runtime
    with pytest.raises(SystemExit) as error:
        cli.main([command,'--inventory',str(inventory),'--output',str(root/'forbidden')])
    assert error.value.code==3
    assert 'G0' in capsys.readouterr().err
    assert not (root/'forbidden').exists()


def test_build_qa_checks_g0_before_target_open(blocked_runtime,monkeypatch):
    import relational64.data as data
    inventory,_=blocked_runtime
    def forbidden(*args,**kwargs):
        raise AssertionError('unapproved target artifacts must not be opened')
    monkeypatch.setattr(data,'TargetArrayStore',forbidden)
    monkeypatch.setattr(cli,'file_hash',lambda path:'0'*64)
    args=cli.parser().parse_args(['build-qa','--inventory',str(inventory)])
    with pytest.raises(ValueError,match='G0'):
        cli.dispatch(args)


@pytest.mark.parametrize('command,extra',[('overfit',[]),('profile',['--stage','physical'])])
def test_explicit_synthetic_routes_never_open_real_data(command,extra,blocked_runtime,monkeypatch):
    import relational64.training as training
    calls=[]
    def fixture(dataset,**kwargs):
        calls.append((dataset,kwargs))
        return {'status':'synthetic_software_fixture'}
    def forbidden(*args,**kwargs):
        raise AssertionError('synthetic CLI attempted real data access')
    monkeypatch.setattr(training,'overfit_small',fixture)
    monkeypatch.setattr(training,'profile_physical',fixture)
    monkeypatch.setattr(cli,'supervised_dataset',forbidden)
    result=cli.dispatch(cli.parser().parse_args([command,*extra,'--synthetic']))
    assert result['status']=='synthetic_software_fixture'
    assert calls[0][0] is None and calls[0][1]['synthetic'] is True
