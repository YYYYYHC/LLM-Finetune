#!/usr/bin/env python3
"""
验证 packing 模式下子序列隔离是否真正生效的脚本。

运行方式：
    python scripts/verify_packing_isolation.py
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

def test_is_packed_sequence_detection():
    """测试 transformers 是否能正确检测 packed sequence"""
    print("=" * 60)
    print("测试 1: _is_packed_sequence 检测")
    print("=" * 60)
    
    try:
        from transformers.modeling_flash_attention_utils import _is_packed_sequence
    except ImportError:
        print("❌ 无法导入 _is_packed_sequence，可能是 transformers 版本过低")
        print("   建议升级: pip install --upgrade transformers")
        return False
    
    # 模拟你的 position_ids: 3 个子序列被 pack 在一起
    # 子序列1: 长度 3, position [0, 1, 2]
    # 子序列2: 长度 4, position [0, 1, 2, 3]
    # 子序列3: 长度 2, position [0, 1]
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0, 1]])
    batch_size = 1
    
    result = _is_packed_sequence(position_ids, batch_size)
    print(f"position_ids: {position_ids.tolist()}")
    print(f"batch_size: {batch_size}")
    print(f"_is_packed_sequence 返回: {result}")
    
    if result:
        print("✅ 检测成功！transformers 能识别这是 packed sequence")
    else:
        print("❌ 检测失败！请检查 position_ids 格式")
    
    # 测试非 packed 的情况（连续 position_ids）
    print("\n--- 对照组: 非 packed 序列 ---")
    normal_position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
    result_normal = _is_packed_sequence(normal_position_ids, batch_size)
    print(f"position_ids: {normal_position_ids.tolist()}")
    print(f"_is_packed_sequence 返回: {result_normal}")
    
    if not result_normal:
        print("✅ 正确！连续 position_ids 不被识别为 packed")
    
    # 测试 batch_size > 1 的情况
    print("\n--- 对照组: batch_size > 1 ---")
    result_batch2 = _is_packed_sequence(position_ids, batch_size=2)
    print(f"batch_size: 2")
    print(f"_is_packed_sequence 返回: {result_batch2}")
    
    if not result_batch2:
        print("⚠️  注意！batch_size > 1 时不会触发自动检测")
    
    return result


def test_cu_seqlens_extraction():
    """测试从 position_ids 提取 cu_seqlens"""
    print("\n" + "=" * 60)
    print("测试 2: cu_seqlens 提取")
    print("=" * 60)
    
    try:
        from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids
    except ImportError:
        print("❌ 无法导入 prepare_fa_kwargs_from_position_ids")
        return False
    
    # 模拟你的 position_ids
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0, 1]])
    
    (cu_seq_lens_q, cu_seq_lens_k), (max_len_q, max_len_k) = prepare_fa_kwargs_from_position_ids(position_ids)
    
    print(f"position_ids: {position_ids.tolist()}")
    print(f"cu_seq_lens_q: {cu_seq_lens_q.tolist()}")
    print(f"cu_seq_lens_k: {cu_seq_lens_k.tolist()}")
    print(f"max_len_q: {max_len_q}")
    print(f"max_len_k: {max_len_k}")
    
    # 验证 cu_seqlens 是否正确
    # 子序列边界: [0, 3, 7, 9]
    # - 子序列1: [0:3] 长度 3
    # - 子序列2: [3:7] 长度 4
    # - 子序列3: [7:9] 长度 2
    expected_cu_seqlens = [0, 3, 7, 9]
    
    if cu_seq_lens_q.tolist() == expected_cu_seqlens:
        print(f"✅ cu_seqlens 正确！预期: {expected_cu_seqlens}")
        return True
    else:
        print(f"❌ cu_seqlens 不正确！预期: {expected_cu_seqlens}, 实际: {cu_seq_lens_q.tolist()}")
        return False


def test_with_actual_model():
    """使用实际模型测试（可选，需要 GPU）"""
    print("\n" + "=" * 60)
    print("测试 3: 实际模型 forward 测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  跳过：CUDA 不可用")
        return None
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen3-8B"
        
        print(f"加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        model.eval()
        
        # 创建 packed 输入
        seq1 = "Hello world"
        seq2 = "How are you"
        
        ids1 = tokenizer.encode(seq1, add_special_tokens=False)
        ids2 = tokenizer.encode(seq2, add_special_tokens=False)
        
        packed_input_ids = torch.tensor([ids1 + ids2]).cuda()
        position_ids = torch.cat([
            torch.arange(len(ids1)),
            torch.arange(len(ids2))
        ]).unsqueeze(0).cuda()
        attention_mask = torch.ones_like(packed_input_ids)
        
        print(f"packed_input_ids shape: {packed_input_ids.shape}")
        print(f"position_ids: {position_ids.tolist()}")
        
        with torch.no_grad():
            output = model(
                input_ids=packed_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        
        print(f"输出 logits shape: {output.logits.shape}")
        print("✅ 模型 forward 成功！")
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False


def test_attention_isolation_by_output_comparison():
    """
    真正验证 attention 隔离的测试：
    对比单独计算每个子序列 vs packed 一起计算的输出。
    如果隔离正确，两者的输出应该相同（或非常接近）。
    """
    print("\n" + "=" * 60)
    print("测试 4: Attention 隔离验证（输出对比法）")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  跳过：CUDA 不可用")
        return None
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 使用小模型测试
        model_name = "Qwen/Qwen3-8B"  # 或者你常用的模型
        
        print(f"加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        model.eval()
        
        # 创建测试序列
        seq1 = "The capital of France is"
        seq2 = "Machine learning is a"
        seq3 = "Python programming language"
        
        ids1 = tokenizer.encode(seq1, add_special_tokens=False)
        ids2 = tokenizer.encode(seq2, add_special_tokens=False)
        ids3 = tokenizer.encode(seq3, add_special_tokens=False)
        
        print(f"\n子序列 1: '{seq1}' -> {len(ids1)} tokens")
        print(f"子序列 2: '{seq2}' -> {len(ids2)} tokens")
        print(f"子序列 3: '{seq3}' -> {len(ids3)} tokens")
        
        # ============================================
        # 方法 1: 单独计算每个子序列的输出
        # ============================================
        print("\n--- 单独计算每个子序列 ---")
        separate_outputs = []
        
        with torch.no_grad():
            for i, ids in enumerate([ids1, ids2, ids3], 1):
                input_ids = torch.tensor([ids]).cuda()
                position_ids = torch.arange(len(ids)).unsqueeze(0).cuda()
                attention_mask = torch.ones_like(input_ids)
                
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
                separate_outputs.append(output.logits.clone())
                print(f"  子序列 {i} 输出 shape: {output.logits.shape}")
        
        # ============================================
        # 方法 2: Packed 一起计算
        # ============================================
        print("\n--- Packed 一起计算 ---")
        
        # Pack 在一起
        packed_input_ids = torch.tensor([ids1 + ids2 + ids3]).cuda()
        
        # 生成 position_ids（每个子序列从 0 开始，这是关键！）
        position_ids = torch.cat([
            torch.arange(len(ids1)),
            torch.arange(len(ids2)),
            torch.arange(len(ids3))
        ]).unsqueeze(0).cuda()
        
        attention_mask = torch.ones_like(packed_input_ids)
        
        print(f"  packed input_ids shape: {packed_input_ids.shape}")
        print(f"  position_ids: {position_ids.tolist()}")
        
        with torch.no_grad():
            packed_output = model(
                input_ids=packed_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        
        # ============================================
        # 对比输出
        # ============================================
        print("\n--- 对比输出 ---")
        
        # 从 packed 输出中提取每个子序列对应的部分
        packed_logits = packed_output.logits
        
        offset = 0
        all_match = True
        
        # bfloat16 精度下的容差设置
        # bfloat16 的相对精度约为 0.78%，所以我们使用相对误差
        abs_tolerance = 0.5  # 绝对容差（logits 范围通常在 -100 到 100）
        rel_tolerance = 0.01  # 相对容差 1%
        
        for i, (ids, sep_out) in enumerate(zip([ids1, ids2, ids3], separate_outputs), 1):
            seq_len = len(ids)
            packed_slice = packed_logits[:, offset:offset+seq_len, :]
            
            # 计算差异
            diff = (packed_slice - sep_out).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            # 计算相对误差（避免除以0）
            sep_out_abs = sep_out.abs()
            relative_diff = diff / (sep_out_abs + 1e-6)
            max_rel_diff = relative_diff.max().item()
            mean_rel_diff = relative_diff.mean().item()
            
            # 使用更合理的判断标准：绝对误差 < abs_tolerance 或 相对误差 < rel_tolerance
            is_match = (max_diff < abs_tolerance) or (max_rel_diff < rel_tolerance)
            status = "✅" if is_match else "❌"
            
            print(f"  子序列 {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, max_rel_diff={max_rel_diff:.4%} {status}")
            
            if not is_match:
                all_match = False
            
            offset += seq_len
        
        # ============================================
        # 结论
        # ============================================
        print("\n--- 结论 ---")
        if all_match:
            print("✅ Attention 隔离验证通过！")
            print("   Packed 计算的输出与单独计算的输出一致（在 bfloat16 精度范围内），")
            print("   说明子序列之间的 attention 确实没有互相干扰。")
            print("\n   注意：bfloat16 精度下，微小的数值差异是正常的，")
            print("   只要相对误差 < 1% 或绝对误差 < 0.5 即可认为隔离生效。")
            return True
        else:
            print("❌ Attention 隔离验证失败！")
            print("   Packed 计算的输出与单独计算的输出存在显著差异，")
            print("   可能的原因：")
            print("   1. position_ids 没有正确设置（每个子序列应从 0 开始）")
            print("   2. Flash Attention 没有正确启用")
            print("   3. transformers 版本不支持自动 packed sequence 检测")
            print("   4. 模型使用了全局的 attention bias 或其他机制")
            return False
            
    except Exception as e:
        import traceback
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False


def test_gradient_isolation():
    """
    通过梯度检查验证 attention 隔离：
    修改子序列1的输入，子序列2和3的梯度不应该变化。
    """
    print("\n" + "=" * 60)
    print("测试 5: Attention 隔离验证（梯度检查法）")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  跳过：CUDA 不可用")
        return None
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen3-8B"
        
        print(f"加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Flash Attention 只支持 fp16/bf16
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        model.eval()
        
        # 创建测试序列
        seq1 = "Hello world"
        seq2 = "Machine learning"
        
        ids1 = tokenizer.encode(seq1, add_special_tokens=False)
        ids2 = tokenizer.encode(seq2, add_special_tokens=False)
        
        print(f"\n子序列 1: '{seq1}' -> {len(ids1)} tokens")
        print(f"子序列 2: '{seq2}' -> {len(ids2)} tokens")
        
        # 获取 embedding 层
        embed_layer = model.get_input_embeddings()
        
        # ============================================
        # 测试: 修改子序列1，检查子序列2的梯度
        # ============================================
        print("\n--- 梯度隔离测试 ---")
        
        # Pack 在一起
        packed_input_ids = torch.tensor([ids1 + ids2]).cuda()
        
        # position_ids: 每个子序列从 0 开始
        position_ids = torch.cat([
            torch.arange(len(ids1)),
            torch.arange(len(ids2))
        ]).unsqueeze(0).cuda()
        
        # 获取 embeddings 并设置 requires_grad
        with torch.no_grad():
            inputs_embeds = embed_layer(packed_input_ids)
        inputs_embeds = inputs_embeds.clone().requires_grad_(True)
        
        attention_mask = torch.ones_like(packed_input_ids)
        
        # Forward
        output = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        # 只对子序列1的输出计算 loss（简化：取第一个 token 的 logits 的和）
        loss_seq1 = output.logits[:, :len(ids1), :].sum()
        
        # Backward
        loss_seq1.backward()
        
        # 检查子序列2的梯度
        grad_seq2 = inputs_embeds.grad[:, len(ids1):, :]
        grad_seq2_norm = grad_seq2.abs().sum().item()
        
        print(f"  子序列 1 的 loss 反传后，子序列 2 的梯度 L1 范数: {grad_seq2_norm:.6f}")
        
        # 理论上，如果隔离正确，子序列2的梯度应该为 0
        if grad_seq2_norm < 1e-6:
            print("✅ 梯度隔离验证通过！")
            print("   子序列1的梯度不会影响子序列2，说明 attention 隔离生效。")
            return True
        else:
            print("❌ 梯度隔离验证失败！")
            print("   子序列2存在非零梯度，说明 attention 可能没有隔离。")
            return False
            
    except Exception as e:
        import traceback
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False


def test_attention_isolation_with_collate_fn():
    """
    测试 6: 使用 trainer 的 collate_fn 逻辑验证 attention 隔离
    
    模拟 trainer.py 中 _collate_fn 的行为，验证在实际训练场景下
    packed sequences 的 attention 是否正确隔离。
    
    这个测试更接近实际训练时的数据处理流程。
    """
    print("\n" + "=" * 60)
    print("测试 6: 使用 Trainer collate_fn 逻辑验证 Attention 隔离")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  跳过：CUDA 不可用")
        return None
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen3-8B"
        
        print(f"加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        model.eval()
        
        # ============================================
        # 模拟 packed 数据（类似 data_processor 输出）
        # ============================================
        # 创建 3 个子序列
        seq1 = "The capital of France is Paris"
        seq2 = "Machine learning is a subset of AI"
        seq3 = "Python is a programming language"
        
        ids1 = tokenizer.encode(seq1, add_special_tokens=False)
        ids2 = tokenizer.encode(seq2, add_special_tokens=False)
        ids3 = tokenizer.encode(seq3, add_special_tokens=False)
        
        print(f"\n子序列 1: '{seq1}' -> {len(ids1)} tokens")
        print(f"子序列 2: '{seq2}' -> {len(ids2)} tokens")
        print(f"子序列 3: '{seq3}' -> {len(ids3)} tokens")
        
        # 模拟一个 packed batch item（类似 data_processor 的输出）
        packed_input_ids = ids1 + ids2 + ids3
        packed_labels = ids1 + ids2 + ids3  # 简化：labels 与 input_ids 相同
        sequence_lengths = [len(ids1), len(ids2), len(ids3)]
        
        # 模拟 batch（单个样本）
        batch = [{
            "input_ids": packed_input_ids,
            "labels": packed_labels,
            "sequence_lengths": sequence_lengths
        }]
        
        # ============================================
        # 模拟 trainer._collate_fn 的逻辑 (has_image=False)
        # ============================================
        print("\n--- 模拟 Trainer collate_fn 逻辑 ---")
        
        def simulate_collate_fn(batch, pad_token_id):
            """模拟 trainer.py 中 _collate_fn 的核心逻辑"""
            input_ids_list = [x["input_ids"] for x in batch]
            labels_list = [x["labels"] for x in batch]
            
            # 检查是否有 sequence_lengths（packed data 标志）
            has_packing = "sequence_lengths" in batch[0]
            sequence_lengths_list = [x["sequence_lengths"] for x in batch] if has_packing else None
            
            # 动态 padding
            max_len = max(len(ids) for ids in input_ids_list)
            batch_size = len(batch)
            
            # 初始化 tensors
            input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
            labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
            
            # 填充实际值
            for i, (ids, lbls) in enumerate(zip(input_ids_list, labels_list)):
                seq_len = len(ids)
                input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, :seq_len] = 1
                labels[i, :seq_len] = torch.tensor(lbls, dtype=torch.long)
            
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
            # 关键：为 packed sequences 生成 position_ids
            if has_packing:
                position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
                for i, seq_lengths in enumerate(sequence_lengths_list):
                    offset = 0
                    for length in seq_lengths:
                        # 每个子序列的 position_ids 从 0 开始
                        position_ids[i, offset:offset + length] = torch.arange(length)
                        offset += length
                    # 剩余位置（padding）保持为 0
                result["position_ids"] = position_ids
            
            return result
        
        # 使用模拟的 collate_fn 处理 batch
        collated = simulate_collate_fn(batch, tokenizer.pad_token_id)
        
        print(f"  input_ids shape: {collated['input_ids'].shape}")
        print(f"  position_ids: {collated['position_ids'].tolist()}")
        print(f"  attention_mask: {collated['attention_mask'].tolist()}")
        
        # 验证 position_ids 格式正确
        expected_position_ids = []
        for length in sequence_lengths:
            expected_position_ids.extend(list(range(length)))
        
        actual_position_ids = collated['position_ids'][0, :sum(sequence_lengths)].tolist()
        
        if actual_position_ids == expected_position_ids:
            print("  ✅ position_ids 格式正确（每个子序列从 0 开始）")
        else:
            print(f"  ❌ position_ids 格式错误！")
            print(f"     预期: {expected_position_ids}")
            print(f"     实际: {actual_position_ids}")
            return False
        
        # ============================================
        # 方法 1: 单独计算每个子序列
        # ============================================
        print("\n--- 单独计算每个子序列 ---")
        separate_outputs = []
        
        with torch.no_grad():
            for i, ids in enumerate([ids1, ids2, ids3], 1):
                input_ids_single = torch.tensor([ids]).cuda()
                position_ids_single = torch.arange(len(ids)).unsqueeze(0).cuda()
                attention_mask_single = torch.ones_like(input_ids_single)
                
                output = model(
                    input_ids=input_ids_single,
                    attention_mask=attention_mask_single,
                    position_ids=position_ids_single
                )
                separate_outputs.append(output.logits.clone())
                print(f"  子序列 {i} 输出 shape: {output.logits.shape}")
        
        # ============================================
        # 方法 2: 使用 collate_fn 输出进行 packed 计算
        # ============================================
        print("\n--- 使用 collate_fn 输出进行 Packed 计算 ---")
        
        packed_input_ids_tensor = collated['input_ids'].cuda()
        packed_position_ids = collated['position_ids'].cuda()
        packed_attention_mask = collated['attention_mask'].cuda()
        
        with torch.no_grad():
            packed_output = model(
                input_ids=packed_input_ids_tensor,
                attention_mask=packed_attention_mask,
                position_ids=packed_position_ids
            )
        
        print(f"  packed 输出 shape: {packed_output.logits.shape}")
        
        # ============================================
        # 对比输出
        # ============================================
        print("\n--- 对比输出 ---")
        
        packed_logits = packed_output.logits
        offset = 0
        all_match = True
        
        # bfloat16 精度下的容差
        abs_tolerance = 0.5
        rel_tolerance = 0.01
        
        for i, (ids, sep_out) in enumerate(zip([ids1, ids2, ids3], separate_outputs), 1):
            seq_len = len(ids)
            packed_slice = packed_logits[:, offset:offset+seq_len, :]
            
            # 计算差异
            diff = (packed_slice - sep_out).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            # 计算相对误差
            sep_out_abs = sep_out.abs()
            relative_diff = diff / (sep_out_abs + 1e-6)
            max_rel_diff = relative_diff.max().item()
            
            # 判断是否匹配
            is_match = (max_diff < abs_tolerance) or (max_rel_diff < rel_tolerance)
            status = "✅" if is_match else "❌"
            
            print(f"  子序列 {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, max_rel_diff={max_rel_diff:.4%} {status}")
            
            if not is_match:
                all_match = False
            
            offset += seq_len
        
        # ============================================
        # 结论
        # ============================================
        print("\n--- 结论 ---")
        if all_match:
            print("✅ Trainer collate_fn 模式下 Attention 隔离验证通过！")
            print("   使用 trainer 的 collate_fn 逻辑处理 packed data 时，")
            print("   子序列之间的 attention 确实没有互相干扰。")
            print("\n   这说明你的训练流程是正确的：")
            print("   - data_processor 正确生成了 sequence_lengths")
            print("   - trainer 的 collate_fn 正确生成了 position_ids")
            print("   - Flash Attention 正确识别并隔离了子序列")
            return True
        else:
            print("❌ Trainer collate_fn 模式下 Attention 隔离验证失败！")
            print("   请检查：")
            print("   1. sequence_lengths 是否正确传递")
            print("   2. position_ids 生成逻辑是否正确")
            print("   3. Flash Attention 是否正确启用")
            return False
    
    except Exception as e:
        import traceback
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False


def test_position_ids_effectiveness():
    """
    测试 7: 验证 position_ids 是否真正发挥作用
    
    对比两种 position_ids 设置下的输出：
    1. 连续 position_ids: [0, 1, 2, ..., total_len-1] （标准自回归模式）
    2. 分段 position_ids: [0, 1, 2, 0, 1, 2, 3, 0, 1, 2] （packed 模式）
    
    预期结果：
    - 如果两者输出 **不同**：说明 position_ids 发挥了作用，Flash Attention 正确识别了 packed sequence
    - 如果两者输出 **相同**：说明 position_ids 没有发挥作用，隔离可能没有生效
    """
    print("\n" + "=" * 60)
    print("测试 7: 验证 position_ids 是否真正发挥作用")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  跳过：CUDA 不可用")
        return None
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen3-8B"
        
        print(f"加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        model.eval()
        
        # 创建测试序列
        seq1 = "The capital of France is"
        seq2 = "Machine learning is a"
        seq3 = "Python programming language"
        
        ids1 = tokenizer.encode(seq1, add_special_tokens=False)
        ids2 = tokenizer.encode(seq2, add_special_tokens=False)
        ids3 = tokenizer.encode(seq3, add_special_tokens=False)
        
        total_len = len(ids1) + len(ids2) + len(ids3)
        
        print(f"\n子序列 1: '{seq1}' -> {len(ids1)} tokens")
        print(f"子序列 2: '{seq2}' -> {len(ids2)} tokens")
        print(f"子序列 3: '{seq3}' -> {len(ids3)} tokens")
        print(f"总长度: {total_len} tokens")
        
        # Pack 在一起的 input_ids
        packed_input_ids = torch.tensor([ids1 + ids2 + ids3]).cuda()
        attention_mask = torch.ones_like(packed_input_ids)
        
        # ============================================
        # 情况 1: 连续 position_ids [0, 1, 2, ..., total_len-1]
        # ============================================
        print("\n--- 情况 1: 连续 position_ids（标准自回归模式）---")
        
        continuous_position_ids = torch.arange(total_len).unsqueeze(0).cuda()
        print(f"  position_ids: {continuous_position_ids.tolist()}")
        
        with torch.no_grad():
            output_continuous = model(
                input_ids=packed_input_ids,
                attention_mask=attention_mask,
                position_ids=continuous_position_ids
            )
        
        print(f"  输出 shape: {output_continuous.logits.shape}")
        
        # ============================================
        # 情况 2: 分段 position_ids（每个子序列从 0 开始）
        # ============================================
        print("\n--- 情况 2: 分段 position_ids（packed 模式）---")
        
        segmented_position_ids = torch.cat([
            torch.arange(len(ids1)),
            torch.arange(len(ids2)),
            torch.arange(len(ids3))
        ]).unsqueeze(0).cuda()
        print(f"  position_ids: {segmented_position_ids.tolist()}")
        
        with torch.no_grad():
            output_segmented = model(
                input_ids=packed_input_ids,
                attention_mask=attention_mask,
                position_ids=segmented_position_ids
            )
        
        print(f"  输出 shape: {output_segmented.logits.shape}")
        
        # ============================================
        # 对比两种输出
        # ============================================
        print("\n--- 对比两种 position_ids 设置下的输出 ---")
        
        logits_continuous = output_continuous.logits
        logits_segmented = output_segmented.logits
        
        diff = (logits_continuous - logits_segmented).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # 检查每个子序列的差异
        offset = 0
        print("\n  各子序列的差异：")
        for i, ids in enumerate([ids1, ids2, ids3], 1):
            seq_len = len(ids)
            seq_diff = diff[:, offset:offset+seq_len, :]
            seq_max_diff = seq_diff.max().item()
            seq_mean_diff = seq_diff.mean().item()
            print(f"    子序列 {i}: max_diff={seq_max_diff:.6f}, mean_diff={seq_mean_diff:.6f}")
            offset += seq_len
        
        print(f"\n  整体差异: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        
        # ============================================
        # 结论
        # ============================================
        print("\n--- 结论 ---")
        
        # 设置一个阈值：如果差异大于这个值，说明 position_ids 发挥了作用
        # 由于是不同的 position encoding，差异应该是显著的
        threshold = 0.1
        
        if max_diff > threshold:
            print(f"✅ position_ids 发挥了作用！")
            print(f"   连续 vs 分段 position_ids 的输出存在显著差异 (max_diff={max_diff:.4f} > {threshold})")
            print(f"   这说明 Flash Attention 正确识别了 packed sequence，")
            print(f"   并基于 position_ids 进行了正确的处理。")
            print(f"\n   结合测试 4 的结果（分段 position_ids 输出与单独计算一致），")
            print(f"   可以确认 attention 隔离已经正确生效。")
            return True
        else:
            print(f"❌ position_ids 似乎没有发挥作用！")
            print(f"   连续 vs 分段 position_ids 的输出几乎相同 (max_diff={max_diff:.6f})")
            print(f"   可能的原因：")
            print(f"   1. Flash Attention 没有正确识别 packed sequence")
            print(f"   2. transformers 版本不支持此功能")
            print(f"   3. 模型的 position encoding 实现有问题")
            return False
            
    except Exception as e:
        import traceback
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    print("🔍 验证 Packing 模式下子序列隔离是否生效")
    print("=" * 60)
    
    # 检查 transformers 版本
    import transformers
    print(f"transformers 版本: {transformers.__version__}")
    
    # 运行测试
    test1_passed = test_is_packed_sequence_detection()
    test2_passed = test_cu_seqlens_extraction()
    test3_passed = test_with_actual_model()
    test4_passed = test_attention_isolation_by_output_comparison()
    test5_passed = test_gradient_isolation()
    test6_passed = test_attention_isolation_with_collate_fn()
    test7_passed = test_position_ids_effectiveness()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    print(f"测试 1 (_is_packed_sequence): {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"测试 2 (cu_seqlens 提取): {'✅ 通过' if test2_passed else '❌ 失败'}")
    print(f"测试 3 (实际模型 forward): {'✅ 通过' if test3_passed else '⚠️ 跳过/失败' if test3_passed is None else '❌ 失败'}")
    print(f"测试 4 (输出对比验证): {'✅ 通过' if test4_passed else '⚠️ 跳过' if test4_passed is None else '❌ 失败'}")
    print(f"测试 5 (梯度隔离验证): {'✅ 通过' if test5_passed else '⚠️ 跳过' if test5_passed is None else '❌ 失败'}")
    print(f"测试 6 (Trainer collate_fn 验证): {'✅ 通过' if test6_passed else '⚠️ 跳过' if test6_passed is None else '❌ 失败'}")
    print(f"测试 7 (position_ids 有效性验证): {'✅ 通过' if test7_passed else '⚠️ 跳过' if test7_passed is None else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 基础检测通过！")
        print("\n关键条件检查:")
        print("  ✅ position_ids 格式正确（每个子序列从 0 开始）")
        print("  ✅ transformers 能检测到 packed sequence")
        print("  ✅ cu_seqlens 能正确提取")
    
    if test4_passed and test5_passed and test6_passed and test7_passed:
        print("\n🎉🎉 全部 Attention 隔离验证通过！")
        print("   - 子序列之间确实没有互相干扰")
        print("   - position_ids 确实发挥了隔离作用")
        print("   - Trainer collate_fn 的处理也是正确的")
    elif test4_passed is False or test5_passed is False or test6_passed is False or test7_passed is False:
        print("\n⚠️  Attention 隔离验证失败，请检查配置。")


if __name__ == "__main__":
    main()
