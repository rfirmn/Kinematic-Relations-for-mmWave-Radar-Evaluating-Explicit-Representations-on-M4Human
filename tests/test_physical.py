import numpy as np
import pytest
import torch
from relational64.physical import physical_loss,window_operators,effective_denominators
from relational64.checkpoint import save_checkpoint,load_checkpoint

def fixture(batch=2):
    torch.manual_seed(11)
    target = torch.randn(batch,64,22,3)
    mask = torch.ones(batch,64,22,dtype=torch.bool)
    mask[0,:,1:] = False
    times = torch.arange(64,dtype=torch.float64).expand(batch,-1)/12
    return target,mask,times

def test_accumulation_exact_denominators_and_backward():
    gt,mask,times = fixture()
    predicted = (gt + .05*torch.randn_like(gt)).requires_grad_()
    w,a = window_operators(times)
    terms = physical_loss(predicted,gt,mask,w,a)
    full = terms.total()
    full.backward()
    grad = predicted.grad.clone()
    predicted.grad = None
    for k in range(2):
        micro = physical_loss(predicted[k:k+1],gt[k:k+1],mask[k:k+1],w[k:k+1],a[k:k+1])
        micro.total(terms.denominator).backward()
    torch.testing.assert_close(grad,predicted.grad,atol=1e-5,rtol=1e-5)
    assert torch.isfinite(grad).all() and grad.norm()>0

def test_root_only_and_no_supervision():
    gt,mask,times = fixture(1)
    p = torch.zeros_like(gt,requires_grad=True)
    w,a = window_operators(times)
    terms = physical_loss(p,gt,mask,w,a)
    assert terms.denominator['relative_velocity']==terms.denominator['bone_length']==0
    assert terms.denominator['root_velocity']>0
    terms.total().backward()
    none = physical_loss(p,gt,torch.zeros_like(mask),w,a)
    assert none.total()==0 and all(v==0 for v in none.denominator.values())

def test_atomic_checkpoint_and_provenance(tmp_path):
    model = torch.nn.Linear(3,3)
    opt = torch.optim.AdamW(model.parameters())
    loss = model(torch.ones(1,3)).sum(); loss.backward(); opt.step()
    original = model.weight.clone()
    path = tmp_path/'last.pt'
    save_checkpoint(path,model,opt,provenance={'seed':42,'split':'abc'},epoch=1,step=1)
    with torch.no_grad(): model.weight.zero_()
    loaded = load_checkpoint(path,model,provenance={'seed':42,'split':'abc'},optimizer=opt)
    torch.testing.assert_close(original,model.weight)
    assert loaded['step']==1
    with pytest.raises(ValueError,match='provenance'):
        load_checkpoint(path,model,provenance={'seed':43,'split':'abc'})
