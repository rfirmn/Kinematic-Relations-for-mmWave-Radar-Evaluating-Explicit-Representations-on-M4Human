"""Observe teacher-forced versus generation prefix logits on an authored diagnostic run."""
import argparse
from pathlib import Path
import torch
from relational64.config import ROOT,resolve,read_json,write_json,file_hash
from relational64.alignment import load_language
from relational64.models import SensorProjector,DescriptorNormalizer,build_descriptors
from relational64.language import collate_spliced
from relational64.qa import canonical_answer


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();config=resolve(ROOT/'configs/relational64/runtime.yaml');runtime=config['runtime']
    from relational64.config import active_artifact_path
    data=active_artifact_path(Path(runtime['derived_root'])/'diagnostic_runs'/args.run_id)
    reports=active_artifact_path(Path(runtime['report_root'])/'diagnostic_runs'/args.run_id)
    checkpoints=active_artifact_path(Path(runtime['checkpoint_root'])/'diagnostic_runs'/args.run_id)
    cache=torch.load(data/'features.diagnostic.pt',map_location='cpu',weights_only=False)
    records=read_json(data/'qa.diagnostic.json')['records'];record=next(r for r in records if r['split']=='val' and r['part']=='root')
    bridge,splicer,tokenizer,dtype=load_language(config,read_json(Path(runtime['report_root'])/'language.lock.json'))
    rows=[]
    with torch.no_grad():
        for condition in ('M','S','F','E','R'):
            state=cache['states'][record['window_uid']];norm=None;descriptor=None
            model=SensorProjector(condition).to(runtime['device'])
            model.load_state_dict(torch.load(checkpoints/condition/'diagnostic.pt',map_location='cpu',weights_only=False)['model']);model.eval()
            if condition!='M':
                artifact=read_json(data/f'descriptor_{condition}.diagnostic.json')
                norm=DescriptorNormalizer(condition,artifact['source_checkpoint_hash'],artifact['train_manifest_hash'])
                norm.load_state_dict({name:torch.as_tensor(artifact[name],dtype=value.dtype) for name,value in norm.state_dict().items()})
                norm.to(runtime['device'])
                descriptor=norm(build_descriptors(state['P'].to(runtime['device']),state['V'].to(runtime['device']),state['available'].to(runtime['device']),condition))
            sensor=model(state['Y'].to(runtime['device']),descriptor).sensor_embeddings[0]
            common=dict(uid=record['qa_uid'],task=record['part'],time_s=state['time_s'].to(runtime['device']),sensor_embeddings=sensor)
            teacher=splicer.build(**common,answer=canonical_answer(record['answer']));prefix=splicer.build(**common)
            teacher_hidden=bridge._hidden_states(collate_spliced([teacher]))[:,teacher.prefix_length-1:teacher.prefix_length]
            prefix_hidden=bridge._hidden_states(collate_spliced([prefix]))[:,-1:]
            integer_mask_hidden=bridge.causal_lm.model(inputs_embeds=prefix.inputs_embeds[None],attention_mask=torch.ones((1,prefix.prefix_length),device=sensor.device,dtype=torch.long),
                position_ids=torch.arange(prefix.prefix_length,device=sensor.device)[None],use_cache=False,return_dict=True).last_hidden_state[:,-1:]
            heads=[bridge._head_logits_float32(h)[0,0] for h in (teacher_hidden,prefix_hidden,integer_mask_hidden)]
            native=bridge.causal_lm.get_output_embeddings()(integer_mask_hidden)[0,0].float()
            target=int(teacher.labels[teacher.prefix_length])
            def prediction(logits):
                token=int(logits.argmax());return {'token_id':token,'text':tokenizer.decode([token]),'gold_first_token_logprob':float(logits.log_softmax(-1)[target])}
            rows.append({'condition':condition,'qa_uid':record['qa_uid'],'prefix_identical':torch.equal(teacher.inputs_embeds[:teacher.prefix_length],prefix.inputs_embeds),
                         'gold_token_id':target,'gold_text':tokenizer.decode([target]),'teacher_fp32_head':prediction(heads[0]),
                         'prefix_fp32_head':prediction(heads[1]),'integer_mask_fp32_head':prediction(heads[2]),'native_generation_head':prediction(native),
                         'teacher_prefix_hidden_max_error':float((teacher_hidden.float()-prefix_hidden.float()).abs().max()),
                         'bool_integer_mask_hidden_max_error':float((prefix_hidden.float()-integer_mask_hidden.float()).abs().max()),
                         'sensor_rms':float(sensor.square().mean().sqrt()),'text_embedding_rms':float(bridge.causal_lm.get_input_embeddings().weight[:1000].float().square().mean().sqrt())})
            probes=[]
            for gain in (0.02622128836810589,0.1):
                candidate=splicer.build(**{**common,'sensor_embeddings':sensor*gain})
                generated=bridge.greedy_generate(candidate,tokenizer,splicer.end_token_id)
                from relational64.qa import parse_answer
                parsed=parse_answer(generated.raw_text,record['part'],truncated=generated.truncated)
                probes.append({'gain':gain,'raw_text':generated.raw_text,'parse_status':parsed.status})
            rows[-1]['inference_only_gain_probe']=probes
            del model,norm,sensor,descriptor,teacher,prefix,teacher_hidden,prefix_hidden,integer_mask_hidden,heads,native
            torch.cuda.empty_cache()
    write_json(reports/'prefix_diagnosis.json',{'run_id':args.run_id,'diagnostic_only':True,'scientific_acceptance':False,'actual_M4Human':True,
                'source_sha256':file_hash(__file__),'features_sha256':file_hash(data/'features.diagnostic.pt'),'rows':rows})
    for row in rows:print(row)


if __name__=='__main__':main()
