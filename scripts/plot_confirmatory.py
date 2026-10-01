"""Plot confirmed raw q trajectories and independent-network paired effects."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hidden_memory.experiment_io import file_hash


def plot(root: Path) -> None:
    summary = json.loads((root/'analysis/summary.json').read_text())
    records = [json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
    grouped = {}
    for record in records:
        if record['status'] != 'completed':
            continue
        h,b = record['network'], record['history']
        response = record['response']
        path = root/'histories'/f'network-{h:02d}-history-{b}'/'response'/f"network-{response['replicate']:02d}"
        samples = []
        for row in response['outcomes']:
            with np.load(path/f"future-{row['future']:02d}.npz") as data:
                samples.append(np.array([[data[f'lambda_{scale}_shock_{z}_q'] for z in (0,1)] for scale in (.5,1.,2.)]))
        grouped.setdefault(h,[]).append(np.mean(samples,axis=0))
    network_curves = np.array([np.mean(grouped[h],axis=0) for h in sorted(grouped)])
    mean = network_curves.mean(axis=0)
    with (root/'analysis/networks.csv').open() as stream:
        network_rows = list(csv.DictReader(stream))
    output = root/'figures'
    output.mkdir(exist_ok=False)
    fig, axes = plt.subplots(1,3,figsize=(11,3.9),layout='constrained')
    t = np.unique(np.r_[0,np.geomspace(1,mean.shape[-1]-1,900).astype(int)])
    for z in (0,1):
        for i,(label,color,style) in enumerate(zip(['Near (0.5)','Sham (1)','Far (2)'],['#009e73','#555555','#d55e00'],['-','--','-.'])):
            axes[z].plot(t,mean[i,z,t],label=label,color=color,ls=style,lw=1.4)
        axes[z].set(xscale='symlog',xlabel='Time (MCS)',ylabel='Contribution rate q',
                    ylim=(max(0., float(np.floor(mean.min()*20)/20)),1.),
                    title=f"({'ab'[z]}) {'Unshocked' if z==0 else 'Shocked: delta = 0.30'}")
        axes[z].legend(frameon=False,fontsize=8)
    effects=np.array([float(r['mean_theta']) for r in network_rows])
    axes[2].scatter(effects,np.arange(len(effects)),s=18,color='#0072b2',label='Network mean')
    lo,hi=summary['network_bootstrap_95_ci']
    axes[2].errorbar(summary['theta'],len(effects)+1,xerr=[[summary['theta']-lo],[hi-summary['theta']]],
                    fmt='D',color='black',capsize=3,label='Mean and 95% CI')
    axes[2].axvline(0,color='gray',ls=':',lw=.9)
    axes[2].axvline(summary['minimum_effect'],color='#d55e00',ls='--',lw=.9,label='Planning threshold 0.02')
    axes[2].set(xlabel='Paired loss difference (near - far)',ylabel='Independent network',title='(c) Network-cluster inference')
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc='outside lower center', ncol=3)
    fig.suptitle(f"E2: H = {summary['effective_networks']}; {summary['paired_futures']} paired futures; q curves are descriptive means",fontsize=10)
    for ext in ('png','pdf','svg'):
        fig.savefig(output/f'confirmatory-main.{ext}',dpi=220)
    plt.close(fig)
    np.savez_compressed(output/'plotted-data.npz',mcs=t,network_q=network_curves[:,:,:,t],network_effects=effects)
    with (output/'mean-q-by-arm.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['lambda', 'mean_q_unshocked', 'mean_q_shocked', 'signed_loss'])
        for i,scale in enumerate((.5,1.,2.)):
            baseline, shocked = mean[i,:,1:].mean(axis=1)
            writer.writerow([scale, baseline, shocked, baseline-shocked])
    (output/'provenance.json').write_text(json.dumps({'script_sha256':file_hash(Path(__file__)),
        'analysis_sha256':file_hash(root/'analysis/summary.json'),
        'curves':'future then history then network means; no temporal smoothing',
        'interval':'fixed whole-network bootstrap percentile 95% interval'},indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path)
    plot(parser.parse_args().campaign)
