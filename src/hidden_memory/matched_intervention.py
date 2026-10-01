"""Comparable suboptimal action-gap subsets; no simulation or outcome-based selection."""

import numpy as np

from .learner import policy
from .payoff import reward_bounds
from .state import State, clone
from .parameters import Parameters
from numpy.typing import NDArray


def gap_candidates(state: State, params: Parameters, action: int) -> tuple[NDArray, NDArray, NDArray]:
    others=[a for a in range(3) if a!=action]
    maximum=state.q[...,others].max(axis=-1)
    gap=maximum-state.q[...,action]
    low,high=reward_bounds(state.graph.g,params)
    low,high=min(0.,low/(1-params.gamma)),max(0.,high/(1-params.gamma))
    eligible=gap>1e-10
    for scale in (.5,1.,2.):
        value=maximum-scale*gap
        eligible &= (value>=low)&(value<=high)&(value<maximum)
    return maximum,gap,eligible


def strata(state: State) -> NDArray[np.int64]:
    greedy=state.q==state.q.max(axis=-1,keepdims=True)
    code=(greedy*np.array([1,2,4])).sum(axis=-1)
    return state.strategy[:,None]*24+np.arange(3)[None,:]*8+code


def matched_rows(state: State, params: Parameters, control: int, caliper: float=.05, l1_tolerance: float=.01) -> tuple[NDArray, NDArray, dict]:
    if control not in (0,1) or not 0<caliper<1 or not 0<=l1_tolerance<1:
        raise ValueError('Require C/D control and finite matching tolerances in range')
    _,pg,pe=gap_candidates(state,params,2)
    _,cg,ce=gap_candidates(state,params,control)
    labels=strata(state).ravel();pg,cg=pg.ravel(),cg.ravel()
    pairs=[]
    for label in sorted(set(labels[pe.ravel()])&set(labels[ce.ravel()])):
        p=list(np.flatnonzero(pe.ravel()&(labels==label)))
        c=list(np.flatnonzero(ce.ravel()&(labels==label)))
        common=set(p)&set(c)
        same=[i for i in sorted(common) if abs(pg[i]-cg[i])/max(pg[i],cg[i])<=caliper]
        pairs.extend((i,i) for i in same)
        removed=set(same)
        p=sorted((i for i in p if i not in removed),key=lambda i:(pg[i],i))
        c=sorted((i for i in c if i not in removed),key=lambda i:(cg[i],i))
        i=j=0
        while i<len(p) and j<len(c):
            a,b=pg[p[i]],cg[c[j]]
            if abs(a-b)/max(a,b)<=caliper:
                pairs.append((p[i],c[j]));i+=1;j+=1
            elif a<b:i+=1
            else:j+=1
    matched_before_l1=len(pairs)
    while pairs:
        differences=np.array([pg[p]-cg[c] for p,c in pairs])
        sums=np.array([sum(pg[p] for p,_ in pairs),sum(cg[c] for _,c in pairs)])
        if abs(sums[0]-sums[1])/max(sums)<=l1_tolerance:break
        direction=np.sign(sums[0]-sums[1])
        pairs.pop(int(np.argmax(direction*differences)))
    p=np.array([a for a,_ in pairs],dtype=np.int64)
    c=np.array([b for _,b in pairs],dtype=np.int64)
    sums=[float(pg[p].sum()),float(cg[c].sum())]
    quantiles=[.1,.25,.5,.75,.9]
    audit={'control_action':control,'p_candidates':int(pe.sum()),'control_candidates':int(ce.sum()),
           'same_row_eligible':int((pe&ce).sum()),'matched_before_l1':matched_before_l1,'matched':len(p),
           'fraction_all_rows':len(p)/(state.graph.n*3),'low_coverage':len(p)/(state.graph.n*3)<.2,
           'gap_sums':sums,'l1_relative_difference':abs(sums[0]-sums[1])/max(sums) if pairs else None,
           'p_gap_quantiles':np.quantile(pg[p],quantiles).tolist() if pairs else [],
           'control_gap_quantiles':np.quantile(cg[c],quantiles).tolist() if pairs else [],
           'same_node_row_pairs':int(sum(a==b for a,b in pairs))}
    return p,c,audit


def edit_subset(state: State, params: Parameters, action: int, rows: NDArray, scale: float) -> State:
    if action not in (0,1,2) or scale not in (.5,1.,2.):
        raise ValueError('Unsupported action or gap scale')
    rows=np.asarray(rows,dtype=np.int64)
    maximum,gap,eligible=gap_candidates(state,params,action)
    if rows.ndim!=1 or len(set(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=eligible.size) or not eligible.ravel()[rows].all():
        raise ValueError('Require distinct jointly legal suboptimal rows')
    branch=clone(state)
    if scale!=1.:
        branch.q.reshape(-1,3)[rows,action]=(maximum-scale*gap).ravel()[rows]
    if not np.array_equal(policy(branch.q,params.epsilon),policy(state.q,params.epsilon)):
        raise AssertionError('Matched intervention changed the full initial policy')
    return branch
