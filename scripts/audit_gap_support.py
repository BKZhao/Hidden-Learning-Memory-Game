"""Outcome-blind C/D matching and training-only natural-value support diagnostics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hidden_memory.checkpoint import load_checkpoint
from hidden_memory.experiment_io import file_hash
from hidden_memory.matched_intervention import gap_candidates,matched_rows,strata


def audit(root:Path,reference:Path,output:Path)->dict:
    output.mkdir(exist_ok=False)
    pools={}
    for rep in range(10):
        state,_,_,_=load_checkpoint(reference/'jobs'/f'E1-p00-r{rep:02d}'/'checkpoints/mcs-100000.npz')
        labels=strata(state).ravel();values=state.q[...,2].ravel()
        for label in np.unique(labels):pools.setdefault(int(label),[]).extend(values[labels==label].tolist())
    bounds={key:np.quantile(values,[.01,.99]) for key,values in pools.items() if len(values)>=50}
    records=[json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
    rows=[]
    for record in sorted(records,key=lambda r:(r['network'],r['history'])):
        h,b=record['network'],record['history']
        cp=root/'histories'/f'network-{h:02d}-history-{b}'/'checkpoints'/f"mcs-{record['training']['selection']['selected_mcs']}.npz"
        state,params,_,meta=load_checkpoint(cp)
        folder=output/f'network-{h:02d}-history-{b}'
        folder.mkdir()
        audits=[]
        for action in (0,1):
            p,c,details=matched_rows(state,params,action,caliper=.05,l1_tolerance=.01)
            np.savez_compressed(folder/f'matched-{action}.npz',p_rows=p,control_rows=c)
            audits.append(details)
            rows.append({'network':h,'history':b,**details})
        maximum,gap,eligible=gap_candidates(state,params,2)
        labels=strata(state)
        support=[]
        for scale in (.5,1.,2.):
            value=maximum-scale*gap if scale!=1 else state.q[...,2]
            assessed=outside=0
            for label,(lo,hi) in bounds.items():
                choose=eligible&(labels==label)
                assessed+=int(choose.sum());outside+=int(((value<lo)|(value>hi))[choose].sum())
            support.append({'lambda':scale,'eligible':int(eligible.sum()),'assessed':assessed,
                            'outside_1_99':outside,'outside_fraction_assessed':outside/assessed if assessed else None})
        (folder/'audit.json').write_text(json.dumps({'checkpoint_checksum':meta['checksum'],'matching':audits,'natural_value_support':support},indent=2)+'\n')
    with (output/'matching.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'scope':'Post-E2 mechanism feasibility; no response effects used in matching',
             'caliper_relative_gap':.05,'maximum_l1_difference':.01,'minimum_rows_to_run':30,
             'reference':str(reference),'reference_results_sha256':file_hash(reference/'results.jsonl'),
             'reference_strata_minimum_rows':50,'natural_support':'P value 1-99 percentiles conditional on public strategy, Q row and greedy set',
             'script_sha256':file_hash(Path(__file__)),
             'matching_by_action':{str(action):{'min_rows':min(r['matched'] for r in rows if r['control_action']==action),
                 'max_rows':max(r['matched'] for r in rows if r['control_action']==action),
                 'eligible_history0_networks':[r['network'] for r in rows if r['control_action']==action and r['history']==0 and r['matched']>=30]}
                 for action in (0,1)}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(audit(args.source,args.reference,args.output),indent=2))
