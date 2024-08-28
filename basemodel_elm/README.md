# basemodel_elm

英语能力、语文能力、数学能力统一大模型刷榜代码

## docker run
```
docker run \
--runtime nvidia \
--gpus '"device=7"' \
-it \
-v /mnt/leaderboard/sunqiao:/data \
-p 8000:80 \
-e MODEL_PATH=/data/models/Qwen2-7B-Instruct \
docker.4pd.io/zhaoxuezhi/basemodel_elm:0.2.4
```

## submit-ceph
```
docker_image: harbor-contest.4pd.io/zhaoxuezhi/basemodel_elm:0.2.4
values:
  env:
    - name: MODE
      value: "ceph"
    - name: MODEL_PATH
      value: "/models/Qwen1.5-7B-Chat"
    - name: FEW_SHOT_PATH
      value: "/fewshot/fewshot.txt"
  nodeSelector:
    contest.4pd.io/accelerator: A100-SXM4-80GB
  resources:
    limits:
      cpu: 8
      memory: 32Gi
      nvidia.com/gpu: 1
    requests:
      cpu: 8
      memory: 32Gi
      nvidia.com/gpu: 1
  readinessProbe:
    httpGet:
      path: /ready
      port: 80 
    initialDelaySeconds: 30 
    periodSeconds: 10 
    timeoutSeconds: 5
leaderboard_options:
  nfs:
    - name: model
      srcRelativePath: zhaoxuezhi/models/Qwen1.5-7B-Chat
      mountPoint: /models/Qwen1.5-7B-Chat
      source: ucloud_juicefs
    - name: fewshot
      srcRelativePath: zhaoxuezhi/fewshot
      mountPoint: /fewshot
      source: ceph_customer
```

## submit-model_hub
```
docker_image: docker.4pd.io/zhaoxuezhi/basemodel_elm:0.2.4
values:
  env:
    - name: MODE
      value: model_hub
  readinessProbe:
    httpGet:
      path: /ready
      port: 80 
    initialDelaySeconds: 30 
    periodSeconds: 10 
    timeoutSeconds: 5
leaderboard_options:
  modelhub_deploy_timeout: 600
  modelhub:
    llm:
      modelBranch: main
      modelRepo: public/qwen2-7b-instruct
      gpuModel: A100-SXM4-80GB
      gpuNum: 1
```

## few shot
```
single_choices
下列词语中，没有错别字的一组是（    ）\n\nA.水墨画  烂竽充数\nB.冰激陵  双龙戏珠\nC.青铜器  没精打采
C
---
single_choices
当今世界正处于百年未有之大变局，中国发展带给世界历史性机遇，继续带着冷战时期的旧思维去看新时代的中国，无异于_________。\n填入画横线部分最恰当的一项是：\nA. 一叶障目\nB. 以偏概全\nC. 刻舟求剑\nD. 自欺欺人
C

```

## prompt template
```
{type}@@@{question}
```