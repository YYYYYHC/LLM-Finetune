# # panorama
# python -m src.inference.vllm_engine \
#     --model_path Qwen/Qwen3-VL-8B-Instruct \
#     --adapter_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_b_c_m_p/8BInstruct_part1-8hd_r128alpha256_bs256Ktokens_lr2e-4-continue/checkpoint-16888 \
#     --mode panorama \
#     --arrow_dir /root/lutaojiang/data/tokenized/panorama_condition_part1-5_res1024x512/batch_0014 \
#     --arrow_offset 0 \
#     --arrow_count 256 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_b_c_m_p/generated_scenes/8BInstruct_part1-8hd_r128alpha256_bs256Ktokens_lr2e-4/checkpoint-16888-p \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 1024x512 

# # caption
# python -m src.inference.vllm_engine \
#     --model_path Qwen/Qwen3-VL-4B-Instruct \
#     --adapter_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_c_p/4BInstruct_part1-8hd_r128alpha256_bs256Ktokens_lr2e-4/checkpoint-4000 \
#     --mode caption \
#     --arrow_dir /root/lutaojiang/data/tokenized/caption_condition_high_density_part1-5_85k/batch_0017 \
#     --arrow_offset 0 \
#     --arrow_count 16 \
#     --output_dir ./outputs/generated_scenes/4BInstruct_part1-8hd_r128alpha256_bs256Ktokens_lr2e-4/checkpoint-4000 \
#     --batch_size 4 \
#     --tensor_parallel_size 4 \
#     --temperature 0.3

# # blueprint
# python -m src.inference.vllm_engine \
#     --model_path Qwen/Qwen3-VL-8B-Instruct \
#     --adapter_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_b_c_p/8BInstruct_part1-8hd_r128alpha256_bs256Ktokens_lr2e-4/checkpoint-13000 \
#     --mode blueprint \
#     --arrow_dir /root/lutaojiang/data/tokenized/blueprint_condition_high_density_part8_43k/batch_0008 \
#     --arrow_offset 0 \
#     --arrow_count 16 \
#     --output_dir ./outputs/generated_scenes/8BInstruct_part1-8hd_r128alpha256_bs256Ktokens_lr2e-4/checkpoint-13000 \
#     --batch_size 4 \
#     --tensor_parallel_size 4 \
#     --temperature 0.3


# # interactive lora
# python -m src.inference.vllm_engine \
#     --model_path Qwen/Qwen3-32B \
#     --adapter_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_caption/32B_part1-8_r128_alpha256_lr1e-4_bs256Ktoken/checkpoint-4000 \
#     --mode text \
#     --output_dir ./outputs/condition_caption/generated_scenes/32B_part1-8_r128_alpha256_lr1e-4_bs256Ktoken/checkpoint-4000 \
#     --tensor_parallel_size 4 \
#     --max_num_seqs 1 \
#     --temperature 0.3

# # # interactive full finetune
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_b_c/8B_full_bs256Ktokens_lr5e-5/final \
#     --mode text \
#     --output_dir ./outputs/condition_b_c/generated_scenes/8B_full_bs256Ktokens_lr5e-5/final \
#     --tensor_parallel_size 4 \
#     --max_num_seqs 1 \
#     --temperature 0.3


# # mv
# python -m src.inference.vllm_engine \
#     --model_path Qwen/Qwen3-VL-8B-Instruct \
#     --adapter_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_b_c_m_p/8BInstruct_part1-8hd_r128alpha256_bs256Ktokens_lr2e-4-continue/checkpoint-16888 \
#     --mode multi_view \
#     --arrow_dir /root/lutaojiang/data/tokenized/mv_condition_part6_res512_num1-10/batch_0005 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_b_c_m_p/generated_scenes/8BInstruct_part1-8hd_r128alpha256_bs256Ktokens_lr2e-4/checkpoint-16888-m \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 512x512 \
#     --arrow_offset 0 \
#     --arrow_count 256



# # f1
# python eval/f1_score.py \
# --pred_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m/generated_scenes/8BInstruct_part1-8hd_r128alpha256_bs256Ktokens_lr2e-4/checkpoint-12000/20251227_230439/json

















# ---------------------------------------------------------------------------
# sh1


# # mv
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr5e-5/checkpoint-20000 \
#     --mode multi_view \
#     --arrow_dir /root/lutaojiang/data/tokenized/mv_condition_part6_res512_num1-10/batch_0005 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/generated_scenes/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr5e-5/checkpoint-20000-m \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 512x512 \
#     --arrow_offset 0 \
#     --arrow_count 256

# # panorama
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr5e-5/checkpoint-20000 \
#     --mode panorama \
#     --arrow_dir /root/lutaojiang/data/tokenized/panorama_condition_part1-5_res1024x512/batch_0014 \
#     --arrow_offset 0 \
#     --arrow_count 256 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/generated_scenes/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr5e-5/checkpoint-20000-p \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 1024x512 



# ---------------------------------------------------------------------------
# 8h20

# # mv
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr1e-5_continue_from13k/checkpoint-21000 \
#     --mode multi_view \
#     --arrow_dir /root/lutaojiang/data/tokenized/mv_condition_part6_res512_num1-10/batch_0005 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/generated_scenes/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr1e-5_continue_from13k/checkpoint-21000-m \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 512x512 \
#     --arrow_offset 0 \
#     --arrow_count 256

# # panorama
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr1e-5_continue_from13k/checkpoint-21000 \
#     --mode panorama \
#     --arrow_dir /root/lutaojiang/data/tokenized/panorama_condition_part1-5_res1024x512/batch_0014 \
#     --arrow_offset 0 \
#     --arrow_count 256 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/generated_scenes/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr1e-5_continue_from13k/checkpoint-21000-p \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 1024x512 







# ---------------------------------------------------------------------------
# a800



# # mv
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr5e-6_continue_from13k/checkpoint-8000 \
#     --mode multi_view \
#     --arrow_dir /root/lutaojiang/data/tokenized/mv_condition_part6_res512_num1-10/batch_0005 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/generated_scenes/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr5e-6_continue_from13k/checkpoint-8000-m \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 512x512 \
#     --arrow_offset 0 \
#     --arrow_count 256

# # panorama
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr5e-6_continue_from13k/checkpoint-8000 \
#     --mode panorama \
#     --arrow_dir /root/lutaojiang/data/tokenized/panorama_condition_part1-5_res1024x512/batch_0014 \
#     --arrow_offset 0 \
#     --arrow_count 256 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/generated_scenes/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr5e-6_continue_from13k/checkpoint-8000-p \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 1024x512 
















# ------------------------------------------------------------------------
# banana

# # banana lora
# python -m src.inference.vllm_engine \
#     --model_path outputs/condition_m_p/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr1e-5_continue_from13k/checkpoint-16000 \
#     --adapter_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_banana/8BInstruct_r128alpha256_bs128Ktokens_lr5e-5_5epochs_with_part6/checkpoint-9000 \
#     --mode multi_view \
#     --arrow_dir /root/lutaojiang/data/tokenized/banana_6-8/batch_0001 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_banana/generated_scenes/8BInstruct_r128alpha256_bs128Ktokens_lr5e-5_5epochs_with_part6/checkpoint-9000 \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 512x512 \
#     --arrow_offset 0 \
#     --arrow_count 256


# # banana full
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_banana/8BInstruct_full_bs128Ktokens_lr5e-6_10epochs/checkpoint-2500 \
#     --mode multi_view \
#     --arrow_dir /root/lutaojiang/data/tokenized/banana_6-8/batch_0001 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_banana/generated_scenes/8BInstruct_full_bs128Ktokens_lr5e-6_10epochs/checkpoint-2500 \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 512x512 \
#     --arrow_offset 0 \
#     --arrow_count 256


# # in-the-wild
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_single_ft/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr1e-5_from-mp-16k/checkpoint-1000 \
#     --mode multi_view \
#     --image_dir /root/lutaojiang/code/LLM_Finetuning/outputs/temp/in-the-wild \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_single_ft/generated_scenes/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr1e-5_from-mp-16k/in-the-wild-ckeckpoint-1000 \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.0 \
#     --image_resolution “512x512” \
#     --arrow_offset 0 \
#     --arrow_count 256















# ------------------------------------------------------------------------
# garment full vision

# # mv v3测试集
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/clothes/8BInstruct_final_full_bs256Ktokens_lr5e-5/final \
#     --mode multi_view \
#     --arrow_dir /root/lutaojiang/data_clothes/GC_vlm_final/batch_0051 \
#     --output_dir /root/zhenyang/generated_results/8BInstruct_final_full_bs256Ktokens_lr5e-5/final \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 512x512 \
#     --arrow_offset 0 \
#     --arrow_count 256 \
#     --eval_mode clothes
#     # --prompt "Based on these multi-view images, generate a garment in JSON format:"  # arrow模式不起作用，手动输入图片时需要使用对应prompt

# # mv nano测试集
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/clothes/8BInstruct_final_nano_full_vision_bs256Ktokens_lr1e-5_10epoch_from-final/checkpoint-2000 \
#     --mode multi_view \
#     --arrow_dir /root/lutaojiang/data_clothes/GC_vlm_final_testset/batch_0000 \
#     --output_dir /root/zhenyang/generated_results/8BInstruct_final_nano_full_vision_bs256Ktokens_lr1e-5_10epoch_from-final/checkpoint-2000 \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution “” \
#     --arrow_offset 0 \
#     --arrow_count 256 \
#     --eval_mode clothes
    # --prompt "Based on these multi-view images, generate a garment in JSON format:"  # arrow模式不起作用，手动输入图片时需要使用对应prompt



# ------------------------------------------------------------------------
# garment lora

# # mv
# python -m src.inference.vllm_engine \
#     --model_path Qwen/Qwen3-VL-4B-Instruct \
#     --adapter_path /root/lutaojiang/code/LLM_Finetuning/outputs/clothes/8BInstruct_final_r128alpha256_bs256Ktokens_lr2e-4/final \
#     --mode multi_view \
#     --arrow_dir /root/lutaojiang/data_clothes/GC_vlm_final/batch_0051 \
#     --output_dir /root/zhenyang/generated_results/8BInstruct_final_r128alpha256_bs256Ktokens_lr2e-4/final \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 512x512 \
#     --arrow_offset 0 \
#     --arrow_count 256 \
#     --eval_mode clothes


# ------------------------------------------------------------------------
# garment wild

# # mv
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/clothes/8BInstruct_final_full_vision_bs256Ktokens_lr1e-5_continue/final/final \
#     --output_dir /root/zhenyang/generated_results/8BInstruct_final_full_vision_bs256Ktokens_lr1e-5_continue/wild_v1_final \
#     --mode multi_view \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 512x512 \
#     --eval_mode clothes \
#     --image_dir /root/zhenyang/datasets/fashion_demo/wild_v1/
#     # --prompt "Based on these multi-view images, generate a garment in JSON format:"  # arrow模式不起作用，手动输入图片时需要使用对应prompt

# # mv
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/clothes/8BInstruct_final_nano_full_vision_bs256Ktokens_lr1e-5_10epoch_from-final/checkpoint-1000 \
#     --output_dir /root/zhenyang/generated_results/8BInstruct_final_nano_full_vision_bs256Ktokens_lr1e-5_10epoch_from-final/wild_v2_checkpoint-1000 \
#     --mode multi_view \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution "" \
#     --eval_mode clothes \
#     --image_dir /root/zhenyang/datasets/fashion_demo/wild_v2/fashion-dataset/fashion-dataset/images_sides
#     # --prompt "Based on these multi-view images, generate a garment in JSON format:"  # arrow模式不起作用，手动输入图片时需要使用对应prompt






















# ------------------------------------------------------------------------
# mv ablation

# # mv
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m_p/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr1e-5_continue_from13k/checkpoint-16000 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/temp/mv_ablation2/output \
#     --mode multi_view \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.3 \
#     --image_resolution 512x512 \
#     --image_dir /root/lutaojiang/code/LLM_Finetuning/outputs/temp/mv_ablation2/input

# # view num ablation
# # # mv
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/condition_single_ft/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr1e-5_from-mp-16k/checkpoint-1000 \
#     --mode multi_view \
#     --arrow_dir /root/lutaojiang/data/tokenized/valid/mv_validset_viewnum1/batch_0000 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_single_ft/generated_scenes/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr1e-5_from-mp-16k/checkpoint-1000-m \
#     --batch_size 256 \
#     --tensor_parallel_size 4 \
#     --temperature 0.01 \
#     --image_resolution 512x512 \
#     --arrow_offset 0 \
#     --arrow_count 256



















# multi-res

# # mv
# python -m src.inference.vllm_engine \
#     --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/conditions_m_multires/8BInstruct_full_vision_bs256Ktokens_lr5e-5/final \
#     --mode multi_view \
#     --arrow_dir /root/lutaojiang/data/tokenized/mv_condition_part6_res512_num1-10/batch_0005 \
#     --output_dir /root/lutaojiang/code/LLM_Finetuning/outputs/conditions_m_multires/generated_scenes/8BInstruct_full_vision_bs256Ktokens_lr5e-5/final-m \
#     --batch_size 256 \
#     --tensor_parallel_size 8 \
#     --temperature 0.01 \
#     --image_resolution 512x512 \
#     --arrow_offset 0 \
#     --arrow_count 256











python -m src.inference.vllm_engine \
    --model_path /root/lutaojiang/code/LLM_Finetuning/outputs/clothes/GC_vlm_final_front/8BInstruct_full_vision_bs256Ktokens_lr5e-5/final-test \
    --mode multi_view \
    --arrow_dir /root/lutaojiang/data_clothes/GC_vlm_final_front/batch_0051 \
    --output_dir /root/zhenyang/generated_results/front_only/8BInstruct_final_full_bs256Ktokens_lr5e-5/final \
    --batch_size 256 \
    --tensor_parallel_size 4 \
    --temperature 0.01 \
    --image_resolution 512x512 \
    --arrow_offset 0 \
    --arrow_count 256 \
    --eval_mode clothes



python ../gpu_test.py > /dev/null 2>&1