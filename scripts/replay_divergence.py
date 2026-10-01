"""Bounded reference-step logs of the first E2 policy/action divergence."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np

from hidden_memory.accelerated import advance_mcs
from hidden_memory.checkpoint import load_checkpoint
from hidden_memory.events import make_tape
from hidden_memory.experiment_io import file_hash,write_json
from hidden_memory.intervention import intervene
from hidden_memory.learner import step
from hidden_memory.runner import run_trajectory
from hidden_memory.shock import apply_shock
from hidden_memory.state import clone


def mask(row: np.ndarray) -> tuple:
    return tuple(np.flatnonzero(row==row.max()).tolist())


def prefix(initial: dict, params, tape: np.ndarray, eligible: np.ndarray, path: Path) -> dict:
    branches={scale:clone(initial[scale]) for scale in (.5,2.)}
    first_policy=first_action=None; logged=0
    with path.open('x') as stream:
        for offset,event in enumerate(tape):
            records={};changed=set()
            for scale,branch in branches.items():
                node=int(event[0]*branch.graph.n);row=int(branch.strategy[node])
                before=branch.q[node,row].copy();visits=branch.visit_count[node,row].copy()
                last=branch.last_update_event[node,row].copy()
                nextmax=branch.q[node].max(axis=-1).copy()
                trace=step(branch,params,event);changed.add((node,row))
                crossing=(2 not in trace.greedy_before and 2 in trace.greedy_after)
                records[str(scale)]={'trace':asdict(trace),'q_row_before':before.tolist(),
                    'q_row_after':branch.q[node,row].tolist(),'visit_count_before':visits.tolist(),
                    'last_update_event_before':last.tolist(),'next_max_before':float(nextmax[trace.action]),
                    'initially_edited_row':bool(eligible[node,row]),
                    'P_entered_greedy':crossing,'crossing_update_action':trace.action if crossing else None}
            new_policy=first_policy is None and any(mask(branches[.5].q[node,row])!=mask(branches[2.].q[node,row]) for node,row in changed)
            new_action=first_action is None and records['0.5']['trace']['action']!=records['2.0']['trace']['action']
            entry={'response_event_zero_based':offset,'records':records}
            if new_policy:first_policy=entry
            if new_action:first_action=entry
            if new_policy or new_action or any(r['initially_edited_row'] and (r['trace']['action']==2 or r['P_entered_greedy']) for r in records.values()):
                stream.write(json.dumps(entry)+'\n');logged+=1
            if first_policy is not None and first_action is not None:break
    checked_events=offset+1
    # Compare all state arrays against an independent accelerated replay of the same prefix.
    for scale,branch in branches.items():
        fast=clone(initial[scale]);advance_mcs(fast,params,tape[:checked_events],False)
        for name in ('q','strategy','counts','visit_count','last_update_event'):
            if not np.array_equal(getattr(branch,name),getattr(fast,name)):
                raise AssertionError(f'Reference/accelerated prefix mismatch: {scale}, {name}')
        if branch.event!=fast.event:raise AssertionError('Event index mismatch')
    return {'checked_events':checked_events,'logged_events':logged,'first_policy':first_policy,
            'first_action':first_action,'reference_accelerated_full_state_equal':True,
            'scope':'First-divergence prefix only; not a full activation-time distribution'}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    config={'networks':[0,1,2,3],'history':0,'future':0,'max_prefix_mcs':1000,'full_replay_mcs':1000,
            'stop_prefix':'Both first near/far policy divergence and actual action divergence found, or maximum reached',
            'source':str(args.source),'script_sha256':file_hash(Path(__file__))}
    write_json(args.output/'config.json',config);shutil.copyfile(__file__,args.output/Path(__file__).name)
    records=[json.loads(s) for s in (args.source/'results.jsonl').read_text().splitlines()]
    results=[]
    for network in config['networks']:
        folder=args.output/f'network-{network:02d}';folder.mkdir()
        record=next(r for r in records if r['network']==network and r['history']==0)
        future=record['response']['outcomes'][0]
        if future['future']!=0:raise ValueError('Expected prespecified future zero')
        state,params,_,meta=load_checkpoint(args.source/'histories'/f'network-{network:02d}-history-0/checkpoints/mcs-100000.npz')
        tape=make_tape(np.random.default_rng(future['event_seed']),state.graph.n*1000)
        nodes=np.asarray(future['shock_nodes'],dtype=np.int64)
        data={'network':network,'checkpoint_checksum':meta['checksum'],'event_seed':future['event_seed'],'shock_nodes':nodes.tolist(),'arms':{}}
        with np.load(args.source/'histories'/f'network-{network:02d}-history-0'/'response'/f"network-{record['response']['replicate']:02d}/future-00.npz") as raw:
            for z in (0,1):
                branches,eligible=intervene(state,params)
                for branch in branches.values():
                    if z:apply_shock(branch,nodes)
                data['arms'][str(z)]=prefix(branches,params,tape,eligible,folder/f'prefix-z{z}.jsonl')
                snapshots={}
                for scale,initial in branches.items():
                    trajectory=run_trajectory(initial,params,tape,backend='numba')
                    for metric in ('q','fP'):
                        values=trajectory[metric]
                        if not np.array_equal(values,raw[f'lambda_{scale}_shock_{z}_{metric}'][:1001]):
                            raise AssertionError('E2 archived response replay mismatch')
                        snapshots[f'lambda_{scale}_{metric}']=values
                    snapshots[f'lambda_{scale}_final_visits']=initial.visit_count
                    snapshots[f'lambda_{scale}_final_q']=initial.q
                np.savez_compressed(folder/f'replay-z{z}.npz',**snapshots)
        data['six_arm_1000_mcs_q_fP_bitwise_match']=True
        write_json(folder/'result.json',data);results.append(data)
        print(json.dumps({'network':network,'prefix_events':{z:a['checked_events'] for z,a in data['arms'].items()},'replay_equal':True}),flush=True)
    write_json(args.output/'summary.json',{'config':config,'results':results})


if __name__=='__main__':main()
