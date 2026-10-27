"""Recheck raw artifacts, write Chinese report, then atomically publish finish.txt."""
import csv
import json
from pathlib import Path
import re
import numpy as np
from common import RUN, ROOT, MODELS, now, write_json, sha256, parse_evaluation
from collect_results import collect

def main():
    rows = collect()  # Reopen raw arrays and verify all 30 episodes in every group.
    state = json.loads((RUN/'reports/run_state.json').read_text())
    preflight = json.loads((RUN/'reports/preflight.json').read_text())
    estimate = json.loads((RUN/'reports/timing_estimate.json').read_text())
    source_snapshot = json.loads((RUN/'reports/source_snapshot.json').read_text())
    resources = json.loads((RUN/'reports/benchmark_resources.json').read_text())
    integrity = {'checked_at':now(),'models':{},'evaluations':{},'files':{}}
    # Raw data and model/environment sources must remain those used for this run.
    for episode in preflight['episodes']:
        path = ROOT/'data/sim_transfer_cube_scripted'/('episode_'+str(episode['id'])+'.hdf5')
        assert sha256(path)==episode['sha256'], 'Input dataset changed during experiment'
    for name,policy,chunk in MODELS:
        model = Path(state['models'][name]['directory'])
        log = (model/'train.stdout.log').read_text()
        assert [int(v) for v in re.findall(r'^Epoch (\d+)$',log,re.M)]==list(range(1000))
        assert len(re.findall(r'^Train loss:',log,re.M))==1000
        assert len(re.findall(r'^Val loss:',log,re.M))==1000
        assert 'Best ckpt, val loss' in log and 'Training finished:' in log
        config = json.loads((model/'training_config.json').read_text())
        source_hashes=config.get('source_sha256',source_snapshot['source_sha256'])
        for filename,digest in source_hashes.items():
            assert sha256(ROOT/filename)==digest, 'Project source changed during experiment: '+filename
        for key,value in {'num_epochs':1000,'seed':0,'batch_size':8,'lr':1e-5,
                          'task_name':'sim_transfer_cube_scripted','temporal_agg':False}.items():
            assert config[key]==value
        if policy=='ACT':
            for key,value in {'chunk_size':chunk,'kl_weight':10,'hidden_dim':512,'dim_feedforward':3200}.items():
                assert config[key]==value
        integrity['models'][name] = dict(status='VERIFIED',epochs=1000,
                                        checkpoint_sha256=sha256(model/'policy_best.ckpt'))
        evaluation = Path(state['evaluations'][name]['directory'])
        ecfg = json.loads((evaluation/'evaluation_config.json').read_text())
        assert ecfg['num_eval_episodes']==30 and ecfg['max_saved_videos']==2
        assert ecfg['source_sha256']==source_hashes
        original=(RUN/'backups/imitate_episodes.py.original').read_text()
        expected=original.replace('    num_rollouts = 50\n','    num_rollouts = config.get("num_eval_episodes", 50)\n').replace(
            '        if save_episode:\n            save_videos(',
            '        if save_episode and rollout_id < config.get("max_saved_videos", 50):\n            save_videos(')
        assert (evaluation/'evaluation_source.py').read_text()==expected
        if ecfg.get('top_camera_only'):
            validation=json.loads((RUN/'reports/camera_filter_validation.json').read_text())
            assert validation['passed'] and all(v['pixels_states_rewards_rng_bitwise_equal'] for v in validation['comparisons'])
            import sys
            sys.path.insert(0,str(ROOT))
            from render_controls import top_only_source
            original_observation,optimized_observation=top_only_source()
            assert (evaluation/'observation_source.original.py').read_text()==original_observation
            assert (evaluation/'observation_source.py').read_text()==optimized_observation
            if ecfg.get('reused_video_directory'):
                prefix=json.loads((evaluation/'legacy_prefix_validation.json').read_text())
                assert prefix['expected_indices']==sorted(prefix['verified_indices'])
                assert {0,1}.issubset(set(prefix['verified_indices']))
                assert prefix['reference_directory']==ecfg['reused_video_directory']
                for i in range(2):
                    assert (evaluation/('video'+str(i)+'.mp4')).resolve()==(Path(ecfg['reused_video_directory'])/('video'+str(i)+'.mp4')).resolve()
        metrics = parse_evaluation(evaluation)
        # Verify videos really decode all 400 frames; filenames alone are insufficient.
        import cv2
        for video in metrics['videos']:
            cap = cv2.VideoCapture(video)
            assert cap.isOpened()
            frames = 0
            while True:
                ok,frame = cap.read()
                if not ok: break
                assert frame.ndim==3 and frame.size>0
                frames+=1
            cap.release(); assert frames==400,(video,frames)
        integrity['evaluations'][name] = dict(status='VERIFIED',episodes=30,successes=metrics['successes'],success_rate=metrics['success_rate'])
    figs = ['E1_algorithm_comparison.png','E3_chunk_size_ablation.png']
    import matplotlib.image as mpimg
    for filename in figs:
        path = RUN/'figures'/filename
        pixels = mpimg.imread(path); assert pixels.size>10000 and np.isfinite(pixels).all()
        integrity['files'][str(path)] = sha256(path)
    indexed = {r['model']:r for r in rows}
    m1,m2,m3,m4 = [indexed[n] for n,_,_ in MODELS]
    report = ['# E1、E3 仿真实验快速复现报告','',f'完成检查时间：{now()}。',
              f'实验目录：`{RUN}`。项目目录：`{ROOT}`。',
              '', '## 环境与数据',
              f"服务器：{preflight['platform']}；GPU：{preflight['gpu']}；PyTorch CUDA runtime：{preflight['torch_cuda']}。",
              f"Python：`{preflight['python']}`；版本：{preflight['python_version']}。",
              f"软件版本：`{json.dumps(preflight['versions'],ensure_ascii=False)}`。",
              f"Git 提交：`{preflight['git_commit']}`。初始工作区见 `reports/initial_git_status.txt` 和 `initial_worktree.diff`。",
              '仅使用 `data/sim_transfer_cube_scripted/` 中的 50 条程序化示范；逐文件完整读取数值及图像并记录 SHA256。未使用人工示范数据。',
              '官方数据加载器按 main() 的种子 1 生成固定 40/10 训练/验证划分，模型训练种子为 0。归一化遵循官方实现，使用全部 50 条数据计算统计量；这包含验证轨迹的信息，是官方代码的复现局限。',
              f"MuJoCo 使用 `{preflight['render_backend']}`。当前为 CPU 软件离屏渲染，评估速度可能较慢。",
              '', '## 参数与训练',
              '四模型均为 batch_size=8、num_epochs=1000、lr=1e-5、seed=0、resnet18、顶部相机。ACT 均为 kl_weight=10、hidden_dim=512、dim_feedforward=3200、4 层编码器、7 层解码器、8 个注意力头；仅 k 不同。CNNMLP 使用官方单步行为克隆。未启用混合精度或时间集成。',
              'M1 只训练一次，共用于 E1 和 E3。独立两轮短训练及 50 轮测速均未计入正式实验。最多三个独立训练进程，资源争用/OOM 时降低并发并在独立目录从头重试，不假设无损续训。',
              f"50 轮 ACT 测速耗时 {estimate['benchmark_seconds']:.2f} 秒，含启动平均 {estimate['seconds_per_epoch_including_startup']:.3f} 秒/轮。测速原始 GPU、显存、CPU 数据见 `logs/resource_monitor.jsonl`。",
              f"测速资源采样汇总：`{json.dumps(resources,ensure_ascii=False)}`。5 秒离散采样不能代表瞬时峰值。",
              f"正式训练墙钟耗时 {state['formal_training_wall_seconds']:.2f} 秒（{state['formal_training_wall_seconds']/3600:.3f} 小时）；包括重试的训练进程时长总和 {state['formal_training_sum_process_seconds']:.2f} 秒。",
              '', '| 模型 | k | 训练状态 | 成功训练耗时(秒) | 评估成功数 | 成功率 | 平均回报 |',
              '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        report.append(f"| {r['model']} | {r['chunk_size']} | {r['training_status']} | {r['training_seconds']:.2f} | {r['successes']}/{r['episodes']} | {r['success_rate']:.2%} | {r['average_return']:.3f} |")
    report += ['', '## E1 算法对比',
               f"ACT(k=100) 成功率 {m1['success_rate']:.2%}，CNNMLP 为 {m2['success_rate']:.2%}；差值 {(m1['success_rate']-m2['success_rate'])*100:.2f} 个百分点。此结论仅适用于本次单种子快速实验。",
               f"图表：`{RUN/'figures'/figs[0]}`。",'', '## E3 动作块长度消融',
               f"ACT k=1、10、100 成功率依次为 {m3['success_rate']:.2%}、{m4['success_rate']:.2%}、{m1['success_rate']:.2%}。同数据划分和其他训练参数下比较，未进行时间集成。",
               f"图表：`{RUN/'figures'/figs[1]}`。",'', '## 原始评估、视频和汇总表',
               '四组均采用各自最佳验证检查点及对应统计量，每组完整评估 30 回合。保持官方评估种子 1000、初始状态分布、动作执行、奖励及成功判据（最高奖励等于 4）。每组仅保留前两个回合视频，保存条件不改变评估循环或统计。',
               '评估主循环源码由官方备份仅参数化回合数与视频保存条件；各组 `evaluation_source.py` 可审计。用户授权加速后，另外通过运行时包装过滤 angle/vis 两个无用相机，保持顶部相机原分辨率。项目训练、模型和环境源文件保持原样；运行时观察方法的完整前后源码分别保存在各组 observation_source.original.py / observation_source.py。']
    if state.get('acceleration'):
        camera=json.loads((RUN/'reports/camera_filter_validation.json').read_text())
        report += ['', '### 评估加速与验证',
                   f"经用户明确授权，四组正式评估采用最多 {state.get('max_parallel_evaluations',3)} 个独立进程并行，每个工作进程 CPU Torch 线程数为 1。每组随机种子、400 步动作、30 回合统计均保持原定义。",
                   f"相机过滤测试覆盖两组固定初始位置、每组 8 步（含 reset 共 18 个观察）：顶部图像逐像素、qpos/qvel、物理状态、奖励和 NumPy RNG 均完全一致。原耗时 {camera['original_seconds']:.3f} 秒，过滤后 {camera['top_only_seconds']:.3f} 秒，实测 {camera['measured_speedup']:.3f} 倍；这是短测试速度，不作为完整评估加速倍数。",
                   '旧 M1 慢速评估中断后保留于 evaluations/M1_ACT_k100_before_acceleration_*，未将旧部分回合计入本次正式成功率。新 M1 从回合 0 重新完成完整 30 回合，并逐项核对旧流程已完成回合的回报、最高奖励和成功判定一致。前两个旧视频复用到新评估目录，关联验证见 legacy_prefix_validation.json；其余组前两个视频为顶部相机单视角。',
                   'EGL 已实际重测，但当前容器未暴露 NVIDIA graphics 能力和 EGL 图形库，也无可用 Docker/容器管理入口。未安装依赖、未改系统驱动或 CUDA；因此本次仍使用 OSMesa softpipe。后续容器侧启用方案见 reports/egl_enablement.md。']
    for name,_,_ in MODELS:
        directory = Path(state['evaluations'][name]['directory'])
        metrics = parse_evaluation(directory)
        report.append(f"- {name}：原始结果 `{directory/'result_policy_best.txt'}`，完整 stdout/stderr 日志位于同目录；奖励阈值达到率：`{metrics['threshold_rates']}`。")
        for i,path in enumerate(metrics['videos']):
            report.append(f"  - 回合 {i}（{'成功' if metrics['highest_rewards'][i]==4 else '失败'}，回报 {metrics['returns'][i]}，最高奖励 {metrics['highest_rewards'][i]}）：`{path}`。")
        outcomes = {metrics['highest_rewards'][i]==4 for i in range(2)}
        if True not in outcomes: report.append('  前两个保存回合没有成功案例视频；未额外补录或代用其他回合。')
        if False not in outcomes: report.append('  前两个保存回合没有失败案例视频；未额外补录或代用其他回合。')
    report += ['','汇总表：`reports/summary.csv`；120 回合明细：`reports/episode_level_results.csv`；配置与校验和清单：`reports/experiment_manifest.csv`。',
               '', '## 与原始设置的差异与局限',
               '本次采用 1000 轮训练、每模型 30 回合评估、训练随机种子 0，属于 E1、E3 快速复现，与论文原始设置存在差异；未将论文成功率填入本次结果。当前仓库 README 的训练示例为 2000 轮，官方评估代码默认为 50 回合。此处以本地官方源码/README 为可核实对照，不声称逐项重建论文完整协议。',
               '不包含人工示范实验、E2、E4 或时间集成实验；不是论文所有实验的完整复现。单随机种子结果不能代表多随机种子统计结论；30 回合样本量有限，成功率有抽样误差。',
               '训练样本使用官方随机起点和每轮遍历方式；1000 轮只表示官方 epoch 循环完成 1000 次，不等于每条轨迹所有 400 个时刻均被训练遍历。',
               '最终检查重读最佳检查点/统计文件、核对 1000 轮日志与完整指标、30 条原始回报和最高奖励、成功率与阈值计数、全部视频帧、两张图及报告。所有必要任务均通过；若存在重试，失败日志和检查点保留于各独立重试目录。']
    report_path = RUN/'reports/final_report.md'
    report_path.write_text('\n'.join(report)+'\n')
    assert report_path.stat().st_size>1000
    for file in ['summary.csv','episode_level_results.csv','experiment_manifest.csv','final_report.md']:
        path = RUN/'reports'/file; assert path.stat().st_size>100
        integrity['files'][str(path)] = sha256(path)
    assert sum(1 for _ in csv.DictReader((RUN/'reports/episode_level_results.csv').open()))==120
    write_json(RUN/'reports/integrity_check.json',integrity)
    finish = RUN/'finish.txt'
    assert not finish.exists(), 'Refuse replacing a prior completion marker'
    text = ['E1_AND_E3_COMPLETED','完成时间: '+now(),'实验根目录: '+str(RUN),
            '项目结果根目录: '+str(ROOT/'E1_results')]
    for name,_,_ in MODELS:
        text.append(name+': 训练 SUCCEEDED (1000 epochs), 最佳检查点和归一化统计 VERIFIED; 评估 COMPLETED (30/30)')
    text += ['E1 图表: '+str(RUN/'figures'/figs[0]),'E3 图表: '+str(RUN/'figures'/figs[1]),
             '最终报告: '+str(report_path),'完整性检查: '+str(RUN/'reports/integrity_check.json')]
    tmp = finish.with_suffix('.txt.tmp')
    with tmp.open('x') as f:
        f.write('\n'.join(text)+'\n'); f.flush()
        import os
        os.fsync(f.fileno())
    tmp.replace(finish)
    print('All checks passed; atomically created',finish,flush=True)

if __name__=='__main__': main()
