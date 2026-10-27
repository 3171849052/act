Traceback (most recent call last):
  File "/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/ACT_v3_results/scripts/pipeline.py", line 206, in main
    save('DATA_VALIDATION');train();wait_for_evaluation();evaluate();save('REPORTING')
  File "/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/ACT_v3_results/scripts/pipeline.py", line 130, in train
    else:finish_worker(name,p,rec,'models')
  File "/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/ACT_v3_results/scripts/pipeline.py", line 105, in finish_worker
    if p.returncode:raise RuntimeError(name+' '+group+' failed; logs preserved at '+rec['directory'])
RuntimeError: cube_transfer_human_seed0 models failed; logs preserved at /XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/ACT_v3_results/models/cube_transfer_human_seed0

未创建完成标志；部分目录保留，权重不是完整优化器恢复点。
