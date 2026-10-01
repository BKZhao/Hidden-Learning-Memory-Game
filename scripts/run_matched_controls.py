"""Locked exploratory P versus C/D gap controls, with network-level replication."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import numpy as np

from hidden_memory.checkpoint import load_checkpoint
from hidden_memory.events import make_tape
from hidden_memory.experiment_io import archive_source, file_hash, write_json
from hidden_memory.matched_intervention import edit_subset
from hidden_memory.runner import run_trajectory
from hidden_memory.shock import apply_shock, sample_shock
from hidden_memory.state import clone


def run_network(task: dict) -> dict:
    folder=Path(task['output'])/f"network-{task['network']:02d}"
    folder.mkdir(exist_ok=False)
    state,params,_,meta=load_checkpoint(Path(task['checkpoint']))
    if meta['checksum']!=task['checkpoint_checksum']:
        raise ValueError('Checkpoint differs from locked matching audit')
    streams=np.random.SeedSequence([task['seed'],task['network']]).spawn(2)
    shock_seed,event_seed=[int(s.generate_state(1,dtype=np.uint64)[0]) for s in streams]
    nodes=sample_shock(state,.3,np.random.default_rng(shock_seed))
    tape=make_tape(np.random.default_rng(event_seed),state.graph.n*task['horizon'])
    edits={'sham':clone(state)}
    for action in task['actions']:
        path=Path(task['matching'])/f'matched-{action}.npz'
        if file_hash(path)!=task['matching_sha256'][str(action)]:
            raise ValueError('Matched rows changed after lock')
        with np.load(path) as matched:
            for name,column,rows in [('P',2,matched['p_rows']),('control',action,matched['control_rows'])]:
                for scale in (.5,2.):
                    edits[f'vs{action}_{name}_{scale}']=edit_subset(state,params,column,rows,scale)
    arrays={}; frozen_arrays={}
    for frozen in ([False,True] if task['frozen'] else [False]):
        target=frozen_arrays if frozen else arrays
        for name,initial in edits.items():
            for z in (0,1):
                branch=clone(initial)
                if z:apply_shock(branch,nodes)
                trajectory=run_trajectory(branch,params,tape,freeze=frozen,backend='numba')
                for metric,values in trajectory.items():
                    target[f'{name}_z{z}_{metric}']=values
                    if frozen and not np.array_equal(values,target[f'sham_z{z}_{metric}']):
                        raise AssertionError('Frozen full-horizon trajectory differs')
        np.savez_compressed(folder/('frozen.npz' if frozen else 'trajectories.npz'),**target)
    comparisons={}
    for action in task['actions']:
        outcomes={}
        for name in ('P','control'):
            losses={str(scale):float(np.mean(arrays[f'vs{action}_{name}_{scale}_z0_q'][1:]-arrays[f'vs{action}_{name}_{scale}_z1_q'][1:])) for scale in (.5,2.)}
            outcomes[name]={'losses':losses,'theta':losses['0.5']-losses['2.0']}
        comparisons[str(action)]={**outcomes,'difference_P_minus_control':outcomes['P']['theta']-outcomes['control']['theta']}
    result={'network':task['network'],'history':0,'future':0,'network_hash':meta['network_hash'],
            'checkpoint_checksum':meta['checksum'],'event_seed':event_seed,'shock_seed':shock_seed,
            'shock_nodes':nodes.tolist(),'comparisons':comparisons,'frozen_passed':task['frozen'],
            'branches':len(edits)*2,'horizon':task['horizon']}
    write_json(folder/'result.json',result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if config['launcher_sha256']!=file_hash(Path(__file__)):
        raise ValueError('Launcher changed after configuration lock')
    root=Path(__file__).resolve().parents[1]
    for name,digest in config['package_sha256'].items():
        if file_hash(root/name)!=digest:raise ValueError(f'Simulation source changed: {name}')
    source=Path(config['source']);audit=Path(config['audit'])
    if file_hash(audit/'summary.json')!=config['audit_summary_sha256']:
        raise ValueError('Matching audit summary changed')
    tasks=[]
    for network in config['networks']:
        match=audit/f'network-{network:02d}-history-0'
        data=json.loads((match/'audit.json').read_text())
        actions=[r['control_action'] for r in data['matching'] if r['matched']>=config['minimum_rows']]
        if not actions:continue
        tasks.append({'network':network,'checkpoint':str(source/'histories'/f'network-{network:02d}-history-0/checkpoints/mcs-100000.npz'),
                      'checkpoint_checksum':data['checkpoint_checksum'],'matching':str(match),'actions':actions,
                      'matching_sha256':{str(a):file_hash(match/f'matched-{a}.npz') for a in actions},
                      'horizon':config['horizon'],'seed':config['root_seed'],'frozen':len(tasks)==0,'output':str(args.output)})
    args.output.mkdir(exist_ok=False)
    shutil.copyfile(args.config,args.output/'config.json');shutil.copyfile(__file__,args.output/Path(__file__).name)
    archive_source(args.output,[],config['workers']);write_json(args.output/'tasks.json',tasks)
    status={'state':'running','total':len(tasks),'completed':0,'failed':0};start=perf_counter()
    write_json(args.output/'status.json',status)
    with ProcessPoolExecutor(config['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run_network,t):t for t in tasks}
        for future in as_completed(futures):
            try:
                result=future.result();status['completed']+=1
            except Exception as error:
                result={'network':futures[future]['network'],'error':repr(error),'traceback':traceback.format_exc()};status['failed']+=1
            with (args.output/'results.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            write_json(args.output/'status.json',status);print(json.dumps(status),flush=True)
    status.update(state='failed' if status['failed'] else 'completed',elapsed_seconds=perf_counter()-start)
    write_json(args.output/'status.json',status)
    if status['failed']:raise SystemExit(1)


if __name__=='__main__':main()
