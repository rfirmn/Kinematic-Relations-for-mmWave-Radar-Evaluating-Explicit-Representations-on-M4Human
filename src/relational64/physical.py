"""Masked physical losses with effective-batch numerator/count reduction."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .kinematics import PARENTS, derivative_operator, differentiate_torch

JOINT_WEIGHTS = (1,2,2,1,4,4,1,8,8,1,8,8,1,2,2,1,2,2,4,4,8,8)
LOSS_WEIGHTS = {'position':1.,'root_velocity':5.,'relative_velocity':10.,'bone_length':2.}

@dataclass
class LossTerms:
    numerator: dict
    denominator: dict

    def total(self, denominators=None):
        den = self.denominator if denominators is None else denominators
        return sum(LOSS_WEIGHTS[k]*n/den[k].clamp_min(1) for k,n in self.numerator.items())

    def detached(self):
        return {k:{'numerator':float(n.detach()),'denominator':float(self.denominator[k].detach())} for k,n in self.numerator.items()}

def window_operators(time_s):
    ops = [derivative_operator(t.detach().cpu().numpy()) for t in time_s]
    weights = torch.as_tensor(np.stack([o.weights for o in ops]),device=time_s.device,dtype=torch.float32)
    available = torch.as_tensor(np.stack([o.time_valid for o in ops]),device=time_s.device)
    return weights,available

def velocity_mask(position_mask, time_available):
    available = torch.zeros_like(position_mask)
    for endpoint in range(6,position_mask.shape[1]):
        available[:,endpoint] = position_mask[:,endpoint-6:endpoint+1].all(1) & time_available[:,endpoint,None]
    return available

def physical_loss(predicted, target, target_mask, operator, time_available) -> LossTerms:
    # Explicitly exit AMP for norms/differences and preserve target isolation.
    with torch.autocast(device_type=predicted.device.type,enabled=False):
        p,gt = predicted.float(),target.detach().float()
        pm = target_mask.bool() & torch.isfinite(gt).all(-1)
        gt = torch.where(pm[...,None],gt,0)
        if not torch.isfinite(p).all():
            raise FloatingPointError('Nonfinite predicted pose')
        v = differentiate_torch(p,operator.float())
        vg = differentiate_torch(gt,operator.float())
        vm = velocity_mask(pm,time_available)
        b,t,j,_ = p.shape
        jw = p.new_tensor(JOINT_WEIGHTS)[None,None,:]
        tw = torch.ones(t,device=p.device)
        tw[54:64] = torch.linspace(1,5,10,device=p.device)
        tw = tw[None,:,None]
        numerator,denominator = {},{}
        def add(name,error,mask,weight,vector):
            error = error.abs().sum(-1) if vector else error.abs()
            weighted_mask = mask.to(p.dtype)*weight
            numerator[name] = (torch.where(mask,error,0)*weight).sum()
            denominator[name] = weighted_mask.sum()*(3 if vector else 1)
        add('position',p-gt,pm,tw*jw,True)
        add('root_velocity',v[:,:,0:1]-vg[:,:,0:1],vm[:,:,0:1],tw,True)
        local_error = (v[:,:,1:]-v[:,:,0:1])-(vg[:,:,1:]-vg[:,:,0:1])
        add('relative_velocity',local_error,vm[:,:,1:] & vm[:,:,0:1],tw*jw[:,:,1:],True)
        parents = torch.tensor(PARENTS[1:],device=p.device)
        bones = torch.linalg.vector_norm(p[:,:,1:]-p[:,:,parents],dim=-1)-torch.linalg.vector_norm(gt[:,:,1:]-gt[:,:,parents],dim=-1)
        add('bone_length',bones,pm[:,:,1:] & pm[:,:,parents],tw,False)
    return LossTerms(numerator,denominator)

def effective_denominators(batches, device):
    # Only masks and time are read; retain no encoder activations across micros.
    totals = None
    for batch in batches:
        target = batch['positions'].to(device)
        mask = batch['target_mask'].to(device)
        times = batch['time_s'].to(device)
        w,a = window_operators(times)
        terms = physical_loss(torch.zeros_like(target),target,mask,w,a)
        if totals is None:
            totals = {k:d.detach() for k,d in terms.denominator.items()}
        else:
            for k,d in terms.denominator.items():
                totals[k] += d.detach()
    return totals
