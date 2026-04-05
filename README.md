```
conda create -n LLM python=3.10
pip install -r requirements.txt
pip install --no-build-isolation flash-attn
```

或者直接使用mirrors.tencent.com/lutaojiang/llm:latest
```
conda activate LLM
pip install --no-build-isolation flash-attn
```


1. 处理数据（预tokenize）
具体参考scripts/prepare_data.sh和lutaojiang/code/LLM_Finetuning/src/data里面的逻辑
数据处理的packing模式：因为我本身的任务，每个样本tokenize后长度差一过大，因此我进行了长短打包合并。如果每个样本长度差不多，可以不管这个。

2. 训练
单节点训练参考train.sh
多节点训练参考train_multi_node.sh
