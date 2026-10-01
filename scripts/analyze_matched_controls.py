"""Recompute exploratory matched effects from raw trajectories and bootstrap networks."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hidden_memory.experiment_io import file_hash,write_json


def analyze(root: Path) -> dict:
    config=json.loads((root/'config.json').read_text());status=json.loads((root/'status.json').read_text())
    if status['state']!='completed' or status['failed']:raise ValueError('Incomplete campaign')
    records=[json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
    if sorted(r['network'] for r in records)!=config['networks']:raise ValueError('Missing or duplicate networks')
    if len({r['network_hash'] for r in records})!=len(records):raise ValueError('Repeated graph')
    rows=[];curves=[]
    for r in sorted(records,key=lambda r:r['network']):
        h=r['network']
        with np.load(root/f'network-{h:02d}/trajectories.npz') as data:
            for a,result in r['comparisons'].items():
                row={'network':h,'control_action':int(a)}
                for name in ('P','control'):
                    losses=[]
                    for scale in (.5,2.):
                        q=[]
                        for z in (0,1):
                            v=data[f'vs{a}_{name}_{scale}_z{z}_q']
                            if v.shape!=(5001,) or not np.isfinite(v).all():raise ValueError('Malformed trajectory')
                            q.append(float(v[1:].mean()));row[f'{name}_{scale}_z{z}_mean_q']=q[-1]
                        losses.append(q[0]-q[1])
                    theta=losses[0]-losses[1]
                    if not np.isclose(theta,result[name]['theta'],atol=1e-14,rtol=0):raise ValueError('Raw theta mismatch')
                    row[f'{name}_theta']=theta
                row['difference']=row['P_theta']-row['control_theta'];rows.append(row)
                curves.append(np.array([[data[f'vs{a}_{name}_{scale}_z{z}_q'] for scale in (.5,2.) for z in (0,1)] for name in ('P','control')]))
        if r['frozen_passed']:
            with np.load(root/f'network-{h:02d}/frozen.npz') as data:
                for key in data.files:
                    name,metric=key.split('_z',1)
                    if not np.array_equal(data[key],data[f'sham_z{metric}']):raise ValueError('Frozen trajectory mismatch')
    output=root/'analysis';output.mkdir(exist_ok=False);shutil.copyfile(__file__,output/Path(__file__).name)
    rng=np.random.default_rng(2026100111);summaries={};bootstrap_arrays={}
    for action in sorted({r['control_action'] for r in rows}):
        selected=[r for r in rows if r['control_action']==action]
        index=rng.integers(0,len(selected),size=(10000,len(selected)))
        summaries[str(action)]={'networks':len(selected),'estimates':{}}
        for key in ('P_theta','control_theta','difference'):
            values=np.array([r[key] for r in selected]);boot=values[index].mean(axis=1)
            summaries[str(action)]['estimates'][key]={'mean':float(values.mean()),'sd_network':float(values.std(ddof=1)),
                                                     'ci95':np.quantile(boot,[.025,.975]).tolist()}
            bootstrap_arrays[f'action{action}_{key}']=boot
        summaries[str(action)]['mean_arm_q']={key:float(np.mean([r[key] for r in selected])) for key in selected[0] if key.endswith('mean_q')}
    with (output/'networks.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'bootstrap.npz',**bootstrap_arrays)
    summary={'scope':'Exploratory matched subset effects; H20/B1/K1; no multiplicity adjustment',
             'bootstrap_seed':2026100111,'bootstrap_draws':10000,'by_control_action':summaries,
             'frozen_networks':[r['network'] for r in records if r['frozen_passed']],
             'elapsed_seconds':status['elapsed_seconds'],'results_sha256':file_hash(root/'results.jsonl')}
    write_json(output/'summary.json',summary)
    mean=np.mean(curves,axis=0)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    labels=['Near, unshocked','Near, shocked','Far, unshocked','Far, shocked']
    for i,title in enumerate(('Matched P subset','Matched D subset')):
        for j,label in enumerate(labels):axes[i].plot(np.arange(1,5001),mean[i,j,1:],label=label,lw=1.2)
        axes[i].set(xscale='log',xlabel='Time (MCS)',ylabel='Contribution q',title=title,ylim=(.5,1.01))
    axes[0].legend(fontsize=8,loc='lower right')
    estimates=summaries['1']['estimates']
    for j,key in enumerate(('P_theta','control_theta','difference')):
        v=estimates[key];lo,hi=v['ci95'];m=v['mean']
        axes[2].errorbar(m*100,j,xerr=np.array([[m-lo],[hi-m]])*100,fmt='o',capsize=4)
    axes[2].axvline(0,color='gray',ls=':');axes[2].set(yticks=[0,1,2],yticklabels=['P theta','D theta','P − D'],xlabel='Paired loss difference (pp)',title='20-network mean and 95% CI')
    for suffix in ('png','pdf','svg'):fig.savefig(output/f'matched-controls.{suffix}',dpi=180)
    plt.close(fig)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);args=p.parse_args();print(json.dumps(analyze(args.root),indent=2))
