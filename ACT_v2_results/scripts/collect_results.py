"""Validate all formal artifacts, report seed statistics, plot, then mark completion."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
from common import ROOT,RUN,MODELS,EVALS,TRAINING_SEEDS,NUM_SEEDS,now,write_json,sha256,check_sources,validate_model

FORMAL_EPISODES=50*len(EVALS)


def write_csv(name,rows,fields):
    path=RUN/'reports'/name
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    os.replace(str(temp),str(path))


def collect(state):
    seeds=[];all_episodes=[];pose_reference=None
    for spec in EVALS:
        rec=state['evaluations'][spec['name']]
        if rec['status']!='SUCCEEDED':continue
        directory=Path(rec['directory'])
        rows=[json.loads(s) for s in (directory/'episodes.jsonl').read_text().splitlines()]
        assert len(rows)==50 and [r['episode'] for r in rows]==list(range(50))
        poses=[r['initial_box_pose'] for r in rows]
        if pose_reference is None:pose_reference=poses
        else:assert poses==pose_reference,'Initial state sequences differ across evaluation configurations'
        metrics=json.loads((directory/'evaluation_metrics.json').read_text())
        config=json.loads((directory/'evaluation_config.json').read_text())
        model=Path(state['models'][spec['model']]['directory'])
        artifact=state['models'][spec['model']]['artifacts']
        assert config['checkpoint_sha256']==artifact['checkpoint_sha256']==sha256(model/'policy_best.ckpt')
        assert config['stats_sha256']==artifact['stats_sha256']==sha256(model/'dataset_stats.pkl')
        assert config['eval_seed']==1000 and config['episodes']==50 and config['steps_per_episode']==400
        assert config['temporal_aggregation']==spec['temporal']
        assert metrics['context_released'] and metrics['renderer']['gl_error']==0
        selected=json.loads((directory/'videos.json').read_text())['selected']
        for label in ['success','failure']:
            eligible=[r['episode'] for r in rows if r['success']==(label=='success')]
            if eligible:
                assert selected[label]['episode']==eligible[0]
                assert (directory/selected[label]['path']).stat().st_size>1000
            else:assert label not in selected
        for row in rows:
            rewards=row['rewards']
            assert len(rewards)==row['steps']==400 and all(math.isfinite(v) and v in [0,1,2,3,4] for v in rewards)
            assert max(rewards)==row['highest_reward'] and sum(rewards)==row['episode_return']
            assert row['success']==(max(rewards)==4)
            all_episodes.append(dict(evaluation=spec['name'],group=spec['group'],chunk_size=spec['chunk'],
                training_seed=spec['seed'],temporal_aggregation=spec['temporal'],episode=row['episode'],steps=400,
                success=int(row['success']),highest_reward=row['highest_reward'],episode_return=row['episode_return'],
                reward_ge_1=int(row['reward_ge_1']),reward_ge_2=int(row['reward_ge_2']),reward_ge_3=int(row['reward_ge_3']),
                touch_cube=int(row['physical_stages']['touch_cube']),lift_cube=int(row['physical_stages']['lift_cube']),
                left_contact=int(row['physical_stages']['left_contact']),
                initial_box_pose=json.dumps(row['initial_box_pose']),elapsed_seconds=row['elapsed_seconds'],
                raw_trajectory=str(directory/('episode_%02d.npz'%row['episode']))))
        row=dict(evaluation=spec['name'],group=spec['group'],chunk_size=spec['chunk'],training_seed=spec['seed'],
            temporal_aggregation=spec['temporal'],episodes=50,successes=sum(r['success'] for r in rows),
            success_rate=sum(r['success'] for r in rows)/50,average_episode_return=sum(r['episode_return'] for r in rows)/50,
            touch_cube_rate=sum(r['physical_stages']['touch_cube'] for r in rows)/50,
            lift_cube_rate=sum(r['physical_stages']['lift_cube'] for r in rows)/50,
            left_contact_rate=sum(r['physical_stages']['left_contact'] for r in rows)/50,
            reward_ge_1_rate=sum(r['highest_reward']>=1 for r in rows)/50,
            reward_ge_2_rate=sum(r['highest_reward']>=2 for r in rows)/50,
            reward_ge_3_rate=sum(r['highest_reward']>=3 for r in rows)/50,
            model_directory=str(model),evaluation_directory=str(directory),
            checkpoint_sha256=config['checkpoint_sha256'],stats_sha256=config['stats_sha256'])
        assert abs(row['success_rate']-metrics['success_rate'])<1e-12
        assert abs(row['average_episode_return']-metrics['average_return'])<1e-12
        seeds.append(row)
    summaries=[]
    for group,k in [('main',100),('ablation',1),('ablation',100),('ablation',400)]:
        rows=[r for r in seeds if r['group']==group and r['chunk_size']==k]
        values=[r['success_rate'] for r in rows]
        summaries.append(dict(group=group,chunk_size=k,temporal_aggregation=group=='main',
            completed_training_seeds=len(rows),expected_training_seeds=NUM_SEEDS,episodes=50*len(rows),
            successes=sum(r['successes'] for r in rows),
            mean_seed_success_rate=statistics.mean(values) if values else '',
            sample_std_seed_success_rate=statistics.stdev(values) if len(values)>1 else '',
            pooled_success_rate=sum(r['successes'] for r in rows)/(50*len(rows)) if rows else '',
            mean_episode_return=statistics.mean(r['average_episode_return'] for r in rows) if rows else '',
            mean_touch_cube_rate=statistics.mean(r['touch_cube_rate'] for r in rows) if rows else '',
            mean_lift_cube_rate=statistics.mean(r['lift_cube_rate'] for r in rows) if rows else '',
            mean_left_contact_rate=statistics.mean(r['left_contact_rate'] for r in rows) if rows else '',
            status='complete' if len(rows)==NUM_SEEDS else 'incomplete'))
    seed_fields=['evaluation','group','chunk_size','training_seed','temporal_aggregation','episodes','successes','success_rate',
        'average_episode_return','touch_cube_rate','lift_cube_rate','left_contact_rate','reward_ge_1_rate','reward_ge_2_rate','reward_ge_3_rate',
        'model_directory','evaluation_directory','checkpoint_sha256','stats_sha256']
    episode_fields=['evaluation','group','chunk_size','training_seed','temporal_aggregation','episode','steps','success',
        'highest_reward','episode_return','reward_ge_1','reward_ge_2','reward_ge_3','touch_cube','lift_cube','left_contact',
        'initial_box_pose','elapsed_seconds','raw_trajectory']
    write_csv('per_seed_results.csv',seeds,seed_fields)
    write_csv('episode_results.csv',all_episodes,episode_fields)
    write_csv('summary.csv',summaries,list(summaries[0]))
    return seeds,all_episodes,summaries


def figures(seeds,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':11,'figure.dpi':150})
    main=sorted((r for r in seeds if r['group']=='main'),key=lambda r:r['training_seed'])
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(['Seed '+str(r['training_seed']) for r in main],[100*r['success_rate'] for r in main],color='#237eab')
    ax.axhline(86,color='#b14a3f',linestyle='--',label='Paper reference: 86% (multiple training seeds)')
    ax.set(ylim=(0,100),ylabel='Success rate (%)',title='ACT k=100, temporal aggregation; 50 episodes / seed')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(str(RUN/'figures/main_success.png'));plt.close(fig)
    ablation=sorted((r for r in summaries if r['group']=='ablation'),key=lambda r:r['chunk_size'])
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar([str(r['chunk_size']) for r in ablation],[100*r['mean_seed_success_rate'] for r in ablation],
        yerr=[100*r['sample_std_seed_success_rate'] for r in ablation] if NUM_SEEDS>1 else None,capsize=5,color='#4a9a78')
    for index,row in enumerate(ablation):
        rates=[100*r['success_rate'] for r in seeds if r['group']=='ablation' and r['chunk_size']==row['chunk_size']]
        positions=np.linspace(index-.08,index+.08,len(rates)) if len(rates)>1 else [index]
        ax.scatter(positions,rates,s=18,c='black',zorder=3)
    ax.set(ylim=(0,100),xlabel='Action chunk length k',ylabel='Success rate (%)',
        title=('No temporal aggregation; mean and sample SD across %d seeds'%NUM_SEEDS) if NUM_SEEDS>1
              else 'No temporal aggregation; seed 0 (no between-seed error bars)')
    fig.tight_layout();fig.savefig(str(RUN/'figures/chunk_ablation.png'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(4);width=.18
    configs=[('main',100,'k100 + TA'),('ablation',1,'k1'),('ablation',100,'k100'),('ablation',400,'k400')]
    for index,(group,k,label) in enumerate(configs):
        rows=[r for r in seeds if r['group']==group and r['chunk_size']==k]
        rates=[statistics.mean(r['reward_ge_1_rate'] for r in rows),statistics.mean(r['reward_ge_2_rate'] for r in rows),
               statistics.mean(r['reward_ge_3_rate'] for r in rows),statistics.mean(r['success_rate'] for r in rows)]
        ax.bar(x+(index-1.5)*width,np.array(rates)*100,width,label=label)
    ax.set_xticks(x);ax.set_xticklabels(['Reward >= 1','Reward >= 2','Reward >= 3','Success (4)'])
    ax.set(ylim=(0,100),ylabel='Episode attainment rate (%)',title='Official reward-stage attainment; %d episodes / setting'%(50*NUM_SEEDS))
    ax.legend(ncol=4,fontsize=9);fig.tight_layout();fig.savefig(str(RUN/'figures/reward_stage_attainment.png'));plt.close(fig)


def integrity(state):
    assert all(state['models'][m['name']]['status']=='SUCCEEDED' for m in MODELS)
    assert all(state['evaluations'][e['name']]['status']=='SUCCEEDED' for e in EVALS)
    check_sources()
    preflight=json.loads((RUN/'reports/preflight.json').read_text())
    config_hashes={};stats_hashes=set();initials={1:[],100:[],400:[]}
    for spec in MODELS:
        directory=Path(state['models'][spec['name']]['directory'])
        validation=validate_model(directory,2000)
        assert validation==state['models'][spec['name']]['artifacts']
        config=json.loads((directory/'training_config.json').read_text())
        for key,value in dict(policy_class='ACT',task_name='sim_transfer_cube_scripted',batch_size=8,num_epochs=2000,
            lr=1e-5,kl_weight=10,hidden_dim=512,dim_feedforward=3200,chunk_size=spec['chunk'],seed=spec['seed'],camera_names=['top']).items():
            assert config[key]==value,(spec['name'],key)
        assert json.loads((directory/'data_split.json').read_text())==preflight['split']
        config_hashes[spec['name']]=validation
        stats_hashes.add(validation['stats_sha256'])
        initials[spec['chunk']].append(config['initialization']['query_embedding_sha256'])
    assert len(stats_hashes)==1,'Normalization artifacts differ unexpectedly'
    assert all(len(set(values))==NUM_SEEDS for values in initials.values()),'Training initialization evidence does not match selected seeds'
    before=json.loads((RUN/'provenance/e1_manifest_before.json').read_text());current={}
    history=ROOT/'E1_results'
    for path in sorted(history.rglob('*')):
        if path.is_symlink():current[str(path.relative_to(history))]=dict(symlink=os.readlink(str(path)))
        elif path.is_file():current[str(path.relative_to(history))]=dict(bytes=path.stat().st_size,sha256=sha256(path))
    assert before==current,'E1 historical files changed; refuse finish marker'
    for item in preflight['dataset']:
        assert sha256(ROOT/'data/sim_transfer_cube_scripted'/('episode_%d.hdf5'%item['id']))==item['sha256']
    assets=json.loads((RUN/'provenance/assets_manifest.json').read_text())
    assert all(sha256(ROOT/name)==digest for name,digest in assets.items())
    return dict(passed=True,time=now(),models=config_hashes,e1_entire_manifest_unchanged=True,
        dataset_50_sha256_unchanged=True,physics_assets_unchanged=True,source_unchanged=True,
        shared_split=True,training_seeds=TRAINING_SEEDS,seed_variability_estimated=NUM_SEEDS>1,
        independent_training_initializations=True if NUM_SEEDS>1 else None,
        renderer='OSMesa / llvmpipe CPU',nvidia_egl_enabled=False,cpu_fallback_user_authorized=True)


def report(state,seeds,summaries,complete):
    comp=json.loads((RUN/'reports/software_renderer_comparison.json').read_text())
    pre=json.loads((RUN/'reports/preflight.json').read_text())
    sensitivity_path=RUN/'reports/software_renderer_policy_sensitivity.json'
    sensitivity=json.loads(sensitivity_path.read_text()) if sensitivity_path.exists() else None
    status='全部正式实验及完整性检查完成' if complete else '实验进行中，以下只统计已完成的正式评估；空项表示未测得'
    lines=['# ACT Transfer Cube V2 复现实验','',status+'。更新时间：'+now(),'','## 实际协议与来源','',
        '本轮只使用官方 ACT、50 条程序化示范、顶部 480×640 RGB。用户为加快进度将正式协议改为训练种子 %s：共 %d 个模型（k=1、100、400），各训练 2000 epochs，%d 组评估、每组 50 回合，总计 %d 回合。k100 的同一个模型复用于时间集成主实验和无时间集成消融，不重复训练。'%(TRAINING_SEEDS,len(MODELS),len(EVALS),FORMAL_EPISODES),
        'seed=0 按最小编号及保留已有工作的固定规则选择，不按成功率挑选。seed=1/2 已有权重、日志和部分评估保留，记录于 run_state.json 的 excluded_models / excluded_evaluations，全部排除正式统计。协议变更前文件保存在 backups/single_seed_scope_*。',
        '固定 batch_size=8、lr=1e-5、kl_weight=10、hidden_dim=512、dim_feedforward=3200、ResNet18。保持官方 ImageNet 预训练骨干，ACT 其余参数重新初始化，不加载 E1 权重。',
        '本地 Git：`'+pre['git_commit']+'`；[官方代码](https://github.com/tonyzhaozh/act)固定提交：`'+pre['official_upstream_commit']+'`。模型、损失、数据加载、训练/评估代码与官方逐文件相同，本机 constants 仅包含路径配置差异。',
        '官方 main() 在加载数据前固定种子 1；正式模型共用相同的 40 条训练、10 条验证轨迹。train_bc 再设置模型训练种子。官方归一化统计覆盖所有 50 条轨迹（包含验证轨迹），保留该行为，不擅自改为训练集统计。划分见 preflight.json 和各模型 data_split.json。',
        '评估保持官方 seed=1000，所有配置 50 回合、每回合 400 步、最高奖励等于 4 判成功；检查各配置初始方块序列一致。没有改变动力学、动作分布或奖励计算。','',
        '## EGL 与 CPU 渲染加速','',
        '**NVIDIA EGL 未启用。** 复查仍缺少 libEGL.so.1、NVIDIA vendor 和图形驱动组件，容器没有 graphics 能力且无平台管理权限。按用户最后明确授权，V2 改用 OSMesa CPU 渲染；训练和推理仍使用 A800 CUDA GPU。原 EGL 阻塞报告、管理员配置和本轮复测日志均保留。',
        '实际 GL_RENDERER：`llvmpipe (LLVM 17.0.6, 256 bits)`，Mesa 24.0.5。库来自 Ubuntu 官方 noble 包，下载元数据和 SHA256 保存在 runtime/packages/package_metadata.json，只解包到 V2 用户目录，没有 apt 安装、系统库覆盖、驱动/CUDA 或 Conda 包变更。',
        'Conda Python DT_RPATH 优先加载旧 OSMesa，隔离试验曾发生崩溃，日志完整保留。最终通过进程内 ctypes 库请求路由加载指定 OSMesa，并预加载其配套 GLAPI；只影响本轮进程，不修改 Python 包。真实 Transfer Cube 多相机/顶部相机验证通过才采用。',
        '两组固定方块初始状态，各比较 reset 和五步动作：同一 LLVMpipe 内顶部图像、qpos/qvel、完整物理状态、奖励及 NumPy/Python/Torch RNG 逐位相同。优化只删除 angle、vis 两次无用渲染，AST 验证其余观测逻辑不变。',
        '软件渲染器比较：旧 softpipe 与新 LLVMpipe 物理状态完全相同，像素 MAE %.4f/255，RMSE %.4f，最大差异 %.0f，通道差异>10 占 %.3f%%。已查看两组 PNG，视角和物体布局正常；边缘存在光栅化差异，不要求逐像素相同。'%(
            comp['mae'],comp['rmse'],comp['max'],100*comp['fraction_channels_difference_gt_10']),
        '短测中顶部路径软件渲染加速 %.2f×；加上顶部相机过滤，相对旧三相机路径为 %.2f×。这不是 NVIDIA EGL 加速，不代表端到端训练或全部策略回合的加速比。实际正式评估耗时逐回合记录。'%(
            comp['softpipe_to_llvmpipe_top_speedup'],comp['softpipe_all_to_llvmpipe_top_speedup']),
        '参考：[Mesa LLVMpipe 文档](https://docs.mesa3d.org/drivers/llvmpipe.html)、[Ubuntu Mesa 包来源](https://archive.ubuntu.com/ubuntu/pool/main/m/mesa/)。','',
        '## 调度、记录和恢复','',
        '训练并发：'+str(state.get('train_parallel','尚在测速'))+'；评估并发上限：'+str(state.get('eval_parallel','尚在测速'))+'。训练先实测 1/2/3 进程，按预热后重叠时段单位墙钟时间内实际完成的总 epoch 数选择，完整墙钟吞吐另记；最多三进程。只有 6 核 CPU 配额，不能根据 GPU 显存单独决定。评估用同一初步模型的单/双进程短测选择 1 或 2。后续按用户要求补测 100 轮的第三进程，结果见 live_parallel_tuning.json。',
        '用户要求尽量并行后，调度器允许已完成模型的评估与剩余训练重叠（训练期间评估进程上限见 run_state.json 的 evaluations_during_training，训练完成后使用全部评估槽位）。调度器交接只替换监督进程，保留正在训练的 worker PID、优化器及所有内存状态；交接证据见 reports/parallel_scheduler_migration.json。parallel_controls.json 可在不重启任务的情况下调整并发上限；降低上限仅限制后续启动，不中断正在运行的模型。',
        '资源记录见 logs/resource_monitor.jsonl。每个正式任务具有独立目录、日志、完整参数和源代码信息，训练每轮追加 train/validation 指标，保存最佳验证及最终检查点和配套 dataset_stats.pkl。模型权重不包含完整优化器状态；中断后保留现场，在新的 attempt 目录从头训练，不冒充无损恢复。',
        '吞吐及冒烟模型只在 tests/ 下，排除在正式模型和全部正式统计之外。当前缩减为单种子后，保留一个尚未完成的训练进程，并允许两个已完成模型的评估并发。','',
        '## 逐种子结果','',
        '| 组别 | k | 训练种子 | TA | 成功/回合 | 成功率 | 平均 return |','| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for r in seeds:
        lines.append('| %s | %d | %d | %s | %d/50 | %.2f%% | %.2f |'%(r['group'],r['chunk_size'],r['training_seed'],
            r['temporal_aggregation'],r['successes'],100*r['success_rate'],r['average_episode_return']))
    if not seeds:lines.append('| 尚无完成的正式评估 | | | | | | |')
    lines+=['','## 正式结果汇总','','| 组别 | k | 已完成种子 | 成功次数/回合 | 成功率（多种子时附样本标准差） |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in summaries:
        value='未完成'
        if r['completed_training_seeds']==NUM_SEEDS:
            value=('%.2f%% ± %.2f%%'%(100*r['mean_seed_success_rate'],100*r['sample_std_seed_success_rate'])) if NUM_SEEDS>1 else '%.2f%%（单种子；种子间标准差不适用）'%(100*r['mean_seed_success_rate'])
        lines.append('| %s | %d | %d/%d | %d/%d | %s |'%(r['group'],r['chunk_size'],r['completed_training_seeds'],NUM_SEEDS,r['successes'],r['episodes'],value))
    lines+=['','当前只有一个训练种子，不能估计种子间变异或样本标准差；CSV 的 sample_std_seed_success_rate 留空，不填 0，消融图不绘制种子间误差棒。50 个评估回合不等于 50 个训练种子。阶段图使用官方奖励阈值达到率；另从接触数据只读记录“左右夹爪任一接触方块”“接触方块且脱离桌面”“左夹爪接触”实际事件比例，CSV 同时给出这些指标。奖励≥2 是奖励阶段代理，不能简单等同于独立几何抬起判据。',
        '','## 与论文和历史结果的比较','',
        '[ACT 论文 Table I](https://arxiv.org/pdf/2304.13705)程序化示范 Transfer Cube 的 86% 是三个训练种子、每种子 50 回合的论文参考值，不是本轮实测。']
    main=summaries[0]
    if main['completed_training_seeds']==NUM_SEEDS:
        lines.append('本轮主结果为 %.2f%%，%d 回合共成功 %d 次；与论文 86%% 相差 %+.2f 个百分点。本轮为单训练种子结果，论文为三个训练种子的平均，统计可靠性不同，不能据此宣称完成多种子复现。训练轮数和软件渲染器等协议及环境条件也需要一并考虑。'%(
            main['mean_seed_success_rate']*100,50*NUM_SEEDS,main['successes'],main['mean_seed_success_rate']*100-86))
    else:lines.append('当前主实验尚未完成，不能给出正式成功率或比较差距。')
    lines+=['E1 的 k100、1000 epochs、30 回合、无时间集成成功率 40% 只作历史参考，未合并到 V2 统计。',
        '','## 视频、示范核查与限制','',
        '每组只保存其实际第一个成功回合及第一个失败回合视频；若未出现某类，videos.json 明确记录缺失，不虚构案例。完整 50 回合逐步奖励和初始状态用于统计，不能用视频选择样本代替总体成功率。',
        '示范 50 文件的 SHA256 与 E1 数据清单逐条相同，动作、qpos/qvel 有限，图像形状/类型正确，官方加载器可读，未修改 HDF5。数据没有直接保存原始初始方块 env_state，未在不能可靠恢复该状态的情况下进行专家成功率判定，也没有把错误回放当作数据失败。',
        'CPU 光栅化与训练数据来源渲染器有输入差异；小样本状态/像素验证不能保证所有策略的成功率不受影响，应在条件允许时独立做 NVIDIA EGL 的匹配对照。',
        '','## 文件与完成状态','',
        'CSV：per_seed_results.csv、episode_results.csv、summary.csv。逐步奖励见各评估目录 episodes.jsonl / episode_XX.npz；检查点、统计和配置在 models/ 下。',
        '图：figures/main_success.png、figures/chunk_ablation.png、figures/reward_stage_attainment.png。',
        '完成条件以用户授权的 CPU 渲染及单训练种子协议为准；绝不创建 NVIDIA EGL 成功标志。只有 %d 个模型、%d 组评估、%d 回合、图表和完整性检查全部通过后，才创建 V2 finish.txt。'%(len(MODELS),len(EVALS),FORMAL_EPISODES),
        '当前 finish.txt 状态：'+('本次最终完整性检查后创建，明确标注 OSMesa CPU。' if complete else '未创建。')]
    if sensitivity:
        lines+=['','补充配对推理诊断：用独立 30-epoch 测速模型，在 12 对 old/new 图像和相同 qpos 下，归一化动作 MAE=%.6f、最大绝对差=%.6f；反归一化后的混合动作维度平均绝对差=%.6f（各维物理量不同，不能统一解释成弧度）。这些是可测的策略输入敏感性，**不能声称切换渲染器不影响动作或成功率**。本诊断没有形成成功率估计，不纳入 V2 正式统计。详见 software_renderer_policy_sensitivity.json。'%(
            sensitivity['normalized_action_mae'],sensitivity['normalized_action_max_abs_difference'],sensitivity['denormalized_action_mae'])]
    (RUN/'reports/final_report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    state=json.loads((RUN/'reports/run_state.json').read_text())
    seeds,episodes,summaries=collect(state)
    if not args.partial:
        assert len(seeds)==len(EVALS) and len(episodes)==FORMAL_EPISODES and all(r['status']=='complete' for r in summaries)
        checked=integrity(state)
        figures(seeds,summaries)
        for name in ['main_success.png','chunk_ablation.png','reward_stage_attainment.png']:
            assert (RUN/'figures'/name).stat().st_size>1000
        checked.update(evaluation_groups=len(EVALS),episodes=FORMAL_EPISODES,matching_initial_state_sequences=True,
                       excluded_model_count=len(state.get('excluded_models',{})),excluded_evaluation_count=len(state.get('excluded_evaluations',{})))
        write_json(RUN/'reports/integrity_check.json',checked)
    report(state,seeds,summaries,not args.partial)
    if not args.partial:
        assert not (RUN/'finish.txt').exists()
        lines=['V2_COMPLETED_WITH_USER_AUTHORIZED_OSMESA_CPU_RENDERING','完成时间: '+now(),
            'NVIDIA EGL: 未启用；采用用户明确授权的 CPU 渲染替代协议',
            'GL_RENDERER: llvmpipe (LLVM 17.0.6, 256 bits)',
            'ACT k=1/100/400 × training_seeds=%s，各 2000 epochs；batch_size=8，lr=1e-5，KL=10，hidden=512，FF=3200，ResNet18，top'%TRAINING_SEEDS,
            '单训练种子；种子间标准差不适用。seed=1/2 历史文件保留，排除正式统计。',
            '最佳验证检查点及配套归一化统计全部有效；k100 模型复用于主实验与消融，无重复训练']
        lines += ['训练 '+m['name']+': SUCCEEDED 2000/2000' for m in MODELS]
        lines += ['评估 '+e['name']+': SUCCEEDED 50/50 × 400 steps' for e in EVALS]
        lines += ['正式评估回合合计: '+str(FORMAL_EPISODES),'报告: '+str(RUN/'reports/final_report.md'),
            '完整性检查: '+str(RUN/'reports/integrity_check.json'),'E1_results 全量文件及 50 条示范 SHA256 保持不变']
        (RUN/'finish.txt').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
