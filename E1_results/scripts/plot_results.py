import csv
from common import RUN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    with (RUN/'reports/summary.csv').open() as f: rows = {r['model']:r for r in csv.DictReader(f)}
    plots = [('E1_algorithm_comparison.png',['M1_ACT_k100','M2_CNNMLP'],['ACT (k=100)','CNNMLP'],'Algorithm','E1: algorithm comparison'),
             ('E3_chunk_size_ablation.png',['M3_ACT_k1','M4_ACT_k10','M1_ACT_k100'],['1','10','100'],'Action chunk length k','E3: ACT chunk size ablation')]
    for filename,names,labels,xlabel,title in plots:
        data = [rows[n] for n in names]
        assert all(int(r['episodes'])==30 for r in data)
        values = [float(r['success_rate'])*100 for r in data]
        fig,ax = plt.subplots(figsize=(7,5))
        bars = ax.bar(labels,values,color=['#377eb8','#ff7f00','#4daf4a'][:len(data)],width=0.6)
        for bar,row,value in zip(bars,data,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+2,
                    f"{row['successes']}/{row['episodes']} ({value:.1f}%)",ha='center',va='bottom')
        ax.set(xlabel=xlabel,ylabel='Success rate (%)',ylim=(0,112),title=title)
        ax.set_yticks([0,20,40,60,80,100]); ax.grid(axis='y',alpha=0.2); ax.set_axisbelow(True)
        fig.text(0.5,0.015,'Scripted demonstrations | 1000 epochs | seed 0 | 30 evaluation episodes',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig(RUN/'figures'/filename,dpi=200); plt.close(fig)
    print('Generated the two requested comparison figures',flush=True)

if __name__=='__main__': main()
