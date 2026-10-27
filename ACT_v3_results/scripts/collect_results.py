"""Report genuine completed V3 tasks; held Insertion tasks never count as zero."""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
from common import ROOT,RUN,ALL_TASKS,now,write_json,sha256,check_sources,validate_model
from v2_dependency import inspect_v2


def save_csv(name,records,fields):
    content=io.StringIO();writer=csv.DictWriter(content,fieldnames=fields)
    writer.writeheader();writer.writerows(records)
    path=RUN/'reports'/name;temporary=path.with_name(path.name+'.%d.tmp'%os.getpid())
    temporary.write_text(content.getvalue());os.replace(str(temporary),str(path))


def collect(state):
    summary=[];episodes=[]
    for spec in ALL_TASKS:
        name=spec['name'];rec=state['evaluations'][name]
        row=dict(name=name,task_name=spec['task_name'],seed=0,episode_len=spec['episode_len'],
            status=rec['status'],episodes='',successes='',success_rate='',average_return='',
            reward_ge_1_rate='',reward_ge_2_rate='',reward_ge_3_rate='',reward_ge_4_rate='',
            paper_reference_success_rate=spec['paper_success_rate'],difference_percentage_points='',
            train_seconds='',eval_seconds='',model_directory=state['models'][name].get('directory',''),
            evaluation_directory=rec.get('directory',''))
        if rec['status']=='SUCCEEDED':
            directory=Path(rec['directory']);model=Path(state['models'][name]['directory'])
            rows=[json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines()]
            assert [r['episode'] for r in rows]==list(range(50))
            config=json.loads((directory/'evaluation_config.json').read_text())
            metrics=json.loads((directory/'evaluation_metrics.json').read_text())
            assert config['task_name']==spec['task_name'] and config['episode_len']==spec['episode_len']
            assert config['steps_per_episode']==spec['episode_len'] and config['temporal_aggregation'] and config['eval_seed']==1000
            artifact=state['models'][name]['artifacts']
            assert config['checkpoint_sha256']==artifact['checkpoint_sha256']==sha256(model/'policy_best.ckpt')
            assert config['stats_sha256']==artifact['stats_sha256']==sha256(model/'dataset_stats.pkl')
            assert metrics['episodes']==50 and metrics['context_released'] and metrics['renderer']['gl_error']==0
            assert 'llvmpipe' in metrics['renderer']['GL_RENDERER'].lower()
            for r in rows:
                assert r['steps']==len(r['rewards'])==spec['episode_len']
                assert all(math.isfinite(x) and x in range(r['max_reward']+1) for x in r['rewards'])
                assert sum(r['rewards'])==r['episode_return'] and max(r['rewards'])==r['highest_reward']
                assert r['success']==(r['highest_reward']==r['max_reward'])
                episodes.append(dict(name=name,task_name=spec['task_name'],seed=0,episode=r['episode'],steps=r['steps'],
                    episode_return=r['episode_return'],highest_reward=r['highest_reward'],success=int(r['success']),
                    reward_ge_1=int(r['highest_reward']>=1),reward_ge_2=int(r['highest_reward']>=2),
                    reward_ge_3=int(r['highest_reward']>=3),reward_ge_4=int(r['highest_reward']>=4),
                    elapsed_seconds=r['elapsed_seconds'],initial_pose=json.dumps(r['initial_box_pose']),
                    raw_trajectory=str(directory/('episode_%02d.npz'%r['episode']))))
            successes=sum(r['success'] for r in rows)
            assert abs(metrics['success_rate']-successes/50)<1e-12
            videos=json.loads((directory/'videos.json').read_text())['selected']
            for label in ['success','failure']:
                candidates=[r['episode'] for r in rows if r['success']==(label=='success')]
                if candidates:
                    assert videos[label]['episode']==candidates[0] and (directory/videos[label]['path']).stat().st_size>1000
                else:assert label not in videos
            row.update(status='COMPLETED',episodes=50,successes=successes,success_rate=successes/50,
                average_return=sum(r['episode_return'] for r in rows)/50,
                difference_percentage_points=(successes/50-spec['paper_success_rate'])*100,
                train_seconds=state['models'][name]['attempts'][-1].get('elapsed_seconds',''),
                eval_seconds=rec['attempts'][-1].get('elapsed_seconds',''))
            row.update({'reward_ge_%d_rate'%n:sum(r['highest_reward']>=n for r in rows)/50 for n in range(1,5)})
            completion=dict(task_name=spec['task_name'],training_epochs=2000,evaluation_episodes=50,
                success_rate=successes/50,model_artifacts=artifact,completed_at=now(),
                note='Task-level completion only; not the all-three-task V3 finish marker')
            write_json(RUN/'reports'/(name+'_completion.json'),completion)
        summary.append(row)
    fields=list(summary[0])
    episode_fields=['name','task_name','seed','episode','steps','episode_return','highest_reward','success',
        'reward_ge_1','reward_ge_2','reward_ge_3','reward_ge_4','elapsed_seconds','initial_pose','raw_trajectory']
    save_csv('summary.csv',summary,fields);save_csv('episode_results.csv',episodes,episode_fields)
    return summary,episodes


def plots(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    actual=[r for r in summary if r['status']=='COMPLETED']
    files=[]
    if not actual:return files
    labels={'cube_transfer_human_seed0':'Cube Human','insertion_scripted_seed0':'Insertion Scripted','insertion_human_seed0':'Insertion Human'}
    fig,ax=plt.subplots(figsize=(7,4))
    x=list(range(len(actual)));ax.bar(x,[100*r['success_rate'] for r in actual],color='#277da8',label='Measured seed 0')
    ax.scatter(x,[100*r['paper_reference_success_rate'] for r in actual],marker='_',s=400,c='#c44d3e',label='Paper reference (3 training seeds)')
    ax.set_xticks(x);ax.set_xticklabels([labels[r['name']] for r in actual]);ax.set_ylim(0,100);ax.set_ylabel('Success rate (%)')
    ax.set_title('Completed V3 tasks only; 50 episodes / task, no seed SD');ax.legend(fontsize=8)
    path=RUN/'figures/task_success_rates.png';path.parent.mkdir(exist_ok=True);fig.tight_layout();fig.savefig(str(path));plt.close(fig);files.append(str(path))
    groups=[('cube_transfer_human_seed0','cube_transfer_human_stages.png'),('insertion','insertion_stages.png')]
    for group,filename in groups:
        rows=[r for r in actual if r['name']==group or (group=='insertion' and r['name'].startswith('insertion'))]
        if not rows:continue
        fig,ax=plt.subplots(figsize=(7,4));width=.7/len(rows)
        for index,r in enumerate(rows):
            positions=[n+(index-(len(rows)-1)/2)*width for n in range(4)]
            ax.bar(positions,[100*r['reward_ge_%d_rate'%n] for n in range(1,5)],width,label=labels[r['name']])
        ax.set_xticks(range(4));ax.set_xticklabels(['Reward >=1','Reward >=2','Reward >=3','Success (4)'])
        ax.set_ylim(0,100);ax.set_ylabel('Episode attainment (%)');ax.legend(fontsize=8)
        path=RUN/'figures'/filename;fig.tight_layout();fig.savefig(str(path));plt.close(fig);files.append(str(path))
    return files


def report(state,summary,complete):
    lines=['# ACT V3 单种子实验报告','', '全部三个任务完成。' if complete else '进行中；当前仅启用 Cube Transfer Human，Insertion 由用户准备数据，暂停自动操作。',
        '更新时间：'+now(),'','训练固定 ACT k=100、seed=0、2000 epochs、batch_size=8、lr=1e-5、KL=10、hidden_dim=512、FF=3200、ResNet18、top RGB。保留官方 ImageNet 预训练骨干，其余参数从头初始化，不加载 V1/V2 检查点。',
        '本轮只使用一个训练种子，不估计或虚构种子间标准差。各任务独立数据、检查点及 dataset_stats.pkl；官方 main() 固定划分种子为 1，各任务使用自己的 40/10 轨迹划分与全 50 条轨迹的官方归一化逻辑。',
        '训练使用 A800 CUDA GPU；仿真使用独立 V3 目录内的 OSMesa LLVMpipe CPU 图形库，环境变量在导入前设置，未修改系统或 Conda 依赖。',
        '按用户最新授权，Cube Human 训练与 V2 评估重叠，以单训练进程和较低 CPU 优先级运行。V3 正式评估等待 V2 的实际计划、状态、完成标志、完整性报告与无在跑正式 worker 检查通过；不假定旧的 9 模型/12 评估计划仍有效。V1/V2 文件仅只读访问。',
        '正式评估每任务 50 回合，全部开启官方 Temporal Ensembling、随机种子 1000；Transfer Cube Human / Insertion Scripted / Insertion Human 分别按 constants.py 使用 400/400/500 步，最高奖励等于环境 max_reward 判成功。',
        '顶部相机过滤按实际启用任务在 V3 独立验证，比较两个初始状态的顶部像素、qpos/qvel、完整物理状态、奖励与 NumPy/Python/Torch RNG；渲染身份和误差见 reports/render_selection.json。保留真实第一例成功和第一例失败视频，缺少某类时如实记录，不用视频选择替代完整 50 回合统计。','',
        '| 任务 | 状态 | 成功/回合 | 实测成功率 | 平均回报 | 论文参考 |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in summary:
        done=r['status']=='COMPLETED'
        lines.append('| %s | %s | %s | %s | %s | %.0f%% |'%(r['task_name'],r['status'],
            '%d/50'%r['successes'] if done else '未测得','%.2f%%'%(r['success_rate']*100) if done else '未测得',
            '%.2f'%r['average_return'] if done else '未测得',r['paper_reference_success_rate']*100))
    lines+=['','[ACT 论文 Table I](https://arxiv.org/pdf/2304.13705)参考值来自三个训练种子、每种子 50 回合；本轮是单种子探索性结果，不能当作论文多种子的完整统计复现。',
        '完整原始奖励、动作和 qpos 在各评估目录的 episodes.jsonl / episode_XX.npz；summary.csv 保留阶段奖励达到率、训练及评估耗时、对应检查点目录。',
        '前一次自动启动因配置键拼写错误在进入训练前失败，现场保留在 models/cube_transfer_human_seed0/；修复后在独立 attempt1 目录从头训练，不把仅有权重当作优化器恢复点。',
        'CPU 渲染器相对于训练数据的光栅化实现可能存在像素差异；相机过滤在同一后端验证为精确等价，不能据此推断不同图形驱动的策略成功率完全不变。',
        'Insertion 的已有下载片段保留，自动下载已按用户要求停止；这些文件不能被认作通过检查的完整示范。',
        '全局 finish.txt 仅在三个任务均完成 2000 轮、三组均实际评估 50 回合、150 条记录、三个图表及完整性检查通过后原子创建。当前 Cube 任务级完成记录不等于 V3 全部完成。']
    (RUN/'reports/final_report.md').write_text('\n\n'.join(lines).replace(' |\n\n|',' |\n|')+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');a=parser.parse_args()
    state=json.loads((RUN/'reports/run_state.json').read_text());summary,episodes=collect(state)
    files=plots(summary)
    if not a.partial:
        dependency=inspect_v2();assert dependency['completed']
        assert len(summary)==3 and all(r['status']=='COMPLETED' for r in summary) and len(episodes)==150
        check_sources();checked={}
        for spec in ALL_TASKS:
            name=spec['name'];checked[name]=validate_model(state['models'][name]['directory'],2000)
            assert checked[name]==state['models'][name]['artifacts']
            pre=json.loads((RUN/'reports/data'/name/'validation.json').read_text());assert pre['passed']
            for f in pre['files']:assert sha256(f['path'])==f['sha256']
        assert len(files)==3 and all(Path(p).stat().st_size>1000 for p in files)
        write_json(RUN/'reports/integrity_check.json',dict(passed=True,time=now(),models=checked,episodes=150,
            v2_completed=True,rendering_backend='OSMesa llvmpipe CPU',training_seed=0))
    report(state,summary,not a.partial)
    if not a.partial:
        assert not (RUN/'finish.txt').exists()
        content=['ACT_V3_COMPLETED','V2 完成验证时间: '+state['v2_completed_verified_at'],
            'V3 开始时间: '+state['started_at'],'V3 完成时间: '+now(),'渲染后端: OSMesa / llvmpipe CPU',
            '训练配置: ACT k100 seed0 2000 epochs, batch8, lr1e-5, KL10, hidden512, FF3200, ResNet18, top']
        content+=['%s: 训练 2000/2000，评估 50/50，成功 %d/50（%.2f%%）'%(r['task_name'],r['successes'],100*r['success_rate']) for r in summary]
        content+=['报告: '+str(RUN/'reports/final_report.md')]
        temp=RUN/'finish.txt.tmp';temp.write_text('\n'.join(content)+'\n');os.replace(str(temp),str(RUN/'finish.txt'))


if __name__=='__main__':main()
