# Sequence Packing with FlashAttention VarLen

## 背景
FlashAttention 2 的 `flash_attn_varlen_func` 可以依赖 position id 的“断点”来自动构造块对角 attention mask，因此我们可以在打包多个子序列时完全跳过 `attention_mask`，避免 CPU 侧 mask 构造的开销。但对 Qwen3-VL 而言，视觉 token 的三维 M-RoPE 位置编码在图像内部保持常数，直接把它当作 varlen 的检测信号会导致每个视觉 token 都被视作独立子序列。本次改动的目标是：

- 仍然预先生成与官方 `Qwen3VLModel.get_rope_index()` 等价的三维位置编码；
- 额外提供一份“线性递增且在子序列开头归零”的 1D position ids，只用于触发 FlashAttention 的 packed-sequence 检测；
- 通过将两份 position ids 叠成四维张量，兼容 HuggingFace 在 `position_ids.shape[0] == 4` 时的处理逻辑（第 0 维作为文本位置，其余 3 维进入 M-RoPE）。

## 关键实现
1. **M-RoPE 序列复刻**  
   `[lutaojiang/code/LLM_Finetuning/src/training/trainer.py#L34-L210](lutaojiang/code/LLM_Finetuning/src/training/trainer.py#L34-L210)` 继续完全复现官方 `get_rope_index()`，保证视觉 token 的三元 `(t, h, w)` 编码与 Qwen3-VL 一致。

2. **VarLen 基线 position ids**  
   新增的 `build_varlen_position_ids()`（见 [lutaojiang/code/LLM_Finetuning/src/training/trainer.py#L143-L170](lutaojiang/code/LLM_Finetuning/src/training/trainer.py#L143-L170)）会对每个子序列重新从 0 计数，子序列之间产生 `position_diff != 1` 的断点，从而让 `masking_utils.find_packed_sequence_indices()` 正确识别边界。若单条样本短于批次最大长度，剩余填充区会被视作额外子序列，与真实 token 隔离。

3. **四维 position ids 组合**  
   在 `collate_fn` 内的 packed 分支（[lutaojiang/code/LLM_Finetuning/src/training/trainer.py#L803-L868](lutaojiang/code/LLM_Finetuning/src/training/trainer.py#L803-L868)）先构造 1D 基线，再根据是否存在图像生成 3D M-RoPE，最终使用 `torch.cat([base.unsqueeze(0), vl_position_ids], dim=0)` 得到形如 `(4, batch, seq_len)` 的张量：
   - 第 0 维：供 FlashAttention 检测子序列边界；
   - 第 1-3 维：仍然是 Qwen3-VL 期望的 `(temporal, height, width)`。
   对于纯文本 VL batch，也会构造 3D 张量（所有维度数值一致）保证接口一致；纯文本模型则直接复用 1D 基线。

4. **无 `attention_mask` 推理路径**  
   当 `has_packing` 为真时，不再传 `attention_mask`，完全依赖上述 position ids 触发 varlen（与原先的设计意图保持一致）。

## 设计依据
- HuggingFace 的 `Qwen3VLTextModel.forward`（官方 `transformers/models/qwen3_vl/modeling_qwen3_vl.py`）在检测到 `position_ids.shape[0] == 4` 时，会把第 0 维作为 `text_position_ids`，其余 3 维进入 RoPE，因此我们可以安全地“借位”提供额外的 varlen 信号，而不影响视觉编码。
- `masking_utils.find_packed_sequence_indices()` 只依赖 `position_ids` 的相邻差值是否为 1，因此通过人为重置序列即可让 FlashAttention 构建块对角 mask，无需改动 transformers 源码。

## 使用与验证建议
1. **数据侧**：确保 `sequence_lengths` 真正记录了每个子序列的 token 数；`pack_sequences()` 会自动生成该字段。
2. **训练侧**：在日志中关注 `VL packing enabled: computed stacked position_ids with shape (4, B, L)`，确认四维 position ids 已被构造。必要时可 `print(batch["position_ids"][0])` 查看断点是否与 `sequence_lengths` 对齐。
3. **一致性检查**：可运行一个含图像的微型 batch，通过 `torch.set_printoptions` 输出 `position_ids[:2, 0, :]`，验证文本维度在子序列之间归零、视觉维度仍与 M-RoPE 逻辑匹配。

## 预期收益
- **显存**：varlen 模式无需 4D attention mask，AutoGrad 前向更轻量。
- **隔离**：不同子序列之间的注意力天然被 block-diagonal mask 切断，可在 packed 训练中达到与多 batch 相同的语义隔离效果。
- **兼容性**：实现完全位于 `trainer.py`，无须修改 transformers upstream，可直接随库升级。
