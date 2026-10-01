"""Exploratory temporal decomposition of existing E2 output; primary endpoint stays fixed."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hidden_memory.experiment_io import file_hash


def decompose(root: Path) -> dict:
    primary = json.loads((root/'analysis/summary.json').read_text())
    records = [json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
    metrics = ['q','fP','welfare','punishment_cost']
    grouped = {}
    for record in records:
        if record['status'] != 'completed':
            continue
        h,b = record['network'],record['history']
        response=record['response']
        folder=root/'histories'/f'network-{h:02d}-history-{b}'/'response'/f"network-{response['replicate']:02d}"
        total=None
        for outcome in response['outcomes']:
            with np.load(folder/f"future-{outcome['future']:02d}.npz") as data:
                values=np.array([[[data[f'lambda_{scale}_shock_{z}_{metric}'] for metric in metrics] for z in (0,1)] for scale in (.5,1.,2.)])
            total=values if total is None else total+values
        grouped.setdefault(h,[]).append(total/len(response['outcomes']))
    networks=np.array([np.mean(grouped[h],axis=0) for h in sorted(grouped)])
    mean=networks.mean(axis=0)
    output=root/'decomposition'
    output.mkdir(exist_ok=False)
    windows=[(1,10),(11,50),(51,100),(101,500),(501,1000),(1001,5000)]
    rows=[]
    for metric_index,metric in enumerate(metrics):
        for start,end in windows:
            means=mean[:,:,metric_index,start:end+1].mean(axis=-1)
            d0=float(means[0,0]-means[2,0]);d1=float(means[0,1]-means[2,1])
            rows.append({'metric':metric,'start_mcs':start,'end_mcs':end,'near_minus_far_unshocked':d0,
                         'near_minus_far_shocked':d1,'paired_difference':d0-d1,
                         'contribution_to_full_window':(d0-d1)*(end-start+1)/5000})
    theta=sum(r['contribution_to_full_window'] for r in rows if r['metric']=='q')
    if not np.isclose(theta,primary['theta'],atol=1e-14,rtol=0):
        raise AssertionError('Window decomposition does not sum to the primary endpoint')
    with (output/'windows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    diagnostics=[]
    for i,scale in enumerate((.5,1.,2.)):
        for z in (0,1):
            q=mean[i,z,0]
            location=int(np.argmin(q[1:])+1)
            diagnostics.append({'lambda':scale,'shock':z,'mean_curve_min_q':float(q[location]),'min_at_mcs':location,
                                'mean_q_1001_5000':float(q[1001:].mean()),'mean_fP_1_5000':float(mean[i,z,1,1:].mean()),
                                'mean_welfare_1_5000':float(mean[i,z,2,1:].mean())})
    summary={'scope':'Post-E2 exploratory decomposition; windows are not new confirmatory endpoints',
             'networks':len(networks),'primary_theta_reconstructed':theta,'mean_curve_diagnostics':diagnostics,
             'primary_sha256':file_hash(root/'analysis/summary.json'),'script_sha256':file_hash(Path(__file__))}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(output/'network-curves.npz',metrics=np.array(metrics),network_ids=np.array(sorted(grouped)),curves=networks)
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    t=np.arange(1,5001)
    for ax,m,label in zip(axes.flat,range(4),['Contribution q','Punisher fraction','Welfare','Punishment cost']):
        d0=mean[0,0,m,1:]-mean[2,0,m,1:];d1=mean[0,1,m,1:]-mean[2,1,m,1:]
        ax.plot(t,d0,label='Unshocked: near - far',lw=1,color='#d55e00')
        ax.plot(t,d1,label='Shocked: near - far',lw=1,color='#0072b2')
        ax.axhline(0,color='gray',ls=':',lw=.7)
        ax.set(xscale='log',xlabel='Time (MCS)',ylabel=label,title=f'({chr(97+m)}) {label}')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('E2 exploratory path decomposition; all 20 networks',fontsize=11)
    for ext in ('png','pdf','svg'):fig.savefig(output/f'temporal-decomposition.{ext}',dpi=220)
    plt.close(fig)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('campaign',type=Path)
    print(json.dumps(decompose(parser.parse_args().campaign),indent=2))
