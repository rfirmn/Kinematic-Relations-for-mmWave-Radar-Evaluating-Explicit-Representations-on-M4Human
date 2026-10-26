import json
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from relational64.config import file_hash,write_json
from relational64.evaluation_pipeline import evaluate_physical_cache,text_token_hash,compare_conditions
from relational64.qa import TASKS,canonical_answer

class FakeCache:
    def __init__(self,windows_hash):
        self.meta={'source':{'windows_hash':windows_hash},'split':'val','identity':'cache'}
        self.item={'uid':'w','time_s':torch.arange(64,dtype=torch.float64)/12,'P':torch.zeros(64,22,3),
                   'V':torch.zeros(64,22,3),'centroid':torch.tensor([1.,2.,0.])}
    def __len__(self):return 1
    def __iter__(self):yield self.item

def test_physical_evaluation_restores_centroid(tmp_path):
    windows=tmp_path/'windows.jsonl'
    windows.write_text(json.dumps({'window_uid':'w','split':'val','subject_id':'1','source_keys':[f'[1, 2, {i}]' for i in range(64)]})+'\n')
    targets=SimpleNamespace(root=tmp_path,read=lambda key:(np.tile([1.,2.,0.],(22,1)).astype('float32'),np.ones(22,bool)))
    write_json(tmp_path/'metadata.json',{'fixture':True})
    report=evaluate_physical_cache(FakeCache(file_hash(windows)),windows,targets,tmp_path/'report')
    assert report['status']=='valid'
    assert report['panels']['all_frames']['global_mpjpe']['mean']==0
    assert report['panels']['task_endpoints']['global_mpjpe']['count']==58*22
    with pytest.raises(ValueError,match='manifest mismatch'):
        evaluate_physical_cache(FakeCache('changed'),windows,targets,tmp_path/'bad')

def test_donor_text_hash_uses_common_grid_and_question():
    tokenizer=SimpleNamespace(encode=lambda text,add_special_tokens: list(text.encode()),
                              apply_chat_template=lambda messages,**kw:'system'+messages[1]['content']+'assistant')
    splicer=SimpleNamespace(tokenizer=tokenizer,template_kwargs={})
    times=(np.arange(64)/12).tolist()
    one=text_token_hash(splicer,'arms',times)
    assert one==text_token_hash(splicer,'arms',times)
    times[7]+=.01
    assert one!=text_token_hash(splicer,'arms',times)
    assert one!=text_token_hash(splicer,'legs',times)
    assert text_token_hash(splicer,'arms',times,text_only=True)==text_token_hash(splicer,'arms',[0.]*64,text_only=True)

def test_comparison_averages_seed_scores_without_adding_clusters(tmp_path):
    refs=[];pred={}
    for subject in ('1','2'):
        for part,task in TASKS.items():
            for label in task['classes']:
                uid=f'{subject}-{part}-{label}'
                refs.append({'qa_uid':uid,'window_uid':uid,'subject_id':subject,'recording_id':subject,'part':part,'answer':label,'reference_status':'defined','split':'val'})
                pred[uid]=canonical_answer(label)
    report=compare_conditions(refs,{'R':[pred,pred,pred],'S':[pred,pred,pred]},tmp_path/'comparison.json')
    assert report['scores']['R']==1 and report['contrasts']['R_minus_S']==0
    assert report['seed_sample_std']['R']==0
    assert report['primary_paired_subject_bootstrap']['cluster_count']==2
    assert report['primary_paired_subject_bootstrap']['interval']==[0,0]
