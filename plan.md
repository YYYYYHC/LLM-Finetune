Plan: Verify Pipeline Works with Parametric Curve Data                                                                                                                                                                                                                                                                                                                      
                                                        
 Task Description

 The project at /home/yhc/LLM_Finetuning-main/ is a production-grade Qwen finetuning framework that converts JSON scene data into tokenized Arrow datasets for training. It supports multiple modes (unconditional, blueprint, caption, panorama, multi-view) with a pipeline: raw JSON → conversation format → tokenization → optional packing → Arrow dataset.

 The user has parametric curve data at /home/yhc/cross_sdf/data/thick_structures_slices/ — 17 3D objects represented as cubic B-spline contours across 50 cross-section slices each. The data is structured as {format, bounding_box_xz, slices: [{y, contours: [{closed, deg, n, cx, cz, k, d}]}]}. File sizes range from 32KB to 535KB (compact JSON: ~3.4K to ~57K
 estimated tokens). The files are organized in subdirectories tt/ (1 file) and tt2/ (16 files) with metadata files (processed.json, categories_done.json) that must be excluded.

 The goal is to verify the existing pipeline works with this contour data by making minimal changes. Samples exceeding the max token length should simply be dropped (the pipeline already handles this).

 Token Budget Reference

 ┌─────────────────┬─────────────────────┬──────────┐
 │     Object      │ Whole (est. tokens) │ Fits 8K? │
 ├─────────────────┼─────────────────────┼──────────┤
 │ cube            │ 3,735               │ Yes      │
 ├─────────────────┼─────────────────────┼──────────┤
 │ twocubes        │ 3,445               │ Yes      │
 ├─────────────────┼─────────────────────┼──────────┤
 │ flatCube        │ 3,796               │ Yes      │
 ├─────────────────┼─────────────────────┼──────────┤
 │ cylinder        │ 4,781               │ Yes      │
 ├─────────────────┼─────────────────────┼──────────┤
 │ 3cylinder       │ 5,886               │ Yes      │
 ├─────────────────┼─────────────────────┼──────────┤
 │ torus           │ 7,016               │ Yes      │
 ├─────────────────┼─────────────────────┼──────────┤
 │ eight           │ 7,185               │ Yes      │
 ├─────────────────┼─────────────────────┼──────────┤
 │ beizi           │ 7,459               │ Yes      │
 ├─────────────────┼─────────────────────┼──────────┤
 │ ok              │ 10,952              │ No       │
 ├─────────────────┼─────────────────────┼──────────┤
 │ aibeizi         │ 14,343              │ No       │
 ├─────────────────┼─────────────────────┼──────────┤
 │ armadillo       │ 18,665              │ No       │
 ├─────────────────┼─────────────────────┼──────────┤
 │ hand            │ 19,422              │ No       │
 ├─────────────────┼─────────────────────┼──────────┤
 │ airplane        │ 25,808              │ No       │
 ├─────────────────┼─────────────────────┼──────────┤
 │ shapenetOBJ1    │ 24,886              │ No       │
 ├─────────────────┼─────────────────────┼──────────┤
 │ air_conditioner │ 37,963              │ No       │
 ├─────────────────┼─────────────────────┼──────────┤
 │ mammoth         │ 12,820              │ No       │
 ├─────────────────┼─────────────────────┼──────────┤
 │ brain           │ 56,661              │ No       │
 └─────────────────┴─────────────────────┴──────────┘

 Steps

 Step 1: Flatten the data into a single directory

 A small script (scripts/prepare_contour_data.py) to:
 - Walk thick_structures_slices/ recursively for *.json
 - Skip metadata files (processed.json, categories_done.json)
 - Copy/symlink the 17 data files into one flat output directory

 This is needed because load_json_files() (src/data/loaders.py:33) only does glob("*.json") — non-recursive.

 Step 2: Add "contour" mode to the converter (~15 lines)

 File: src/data/converters.py — add after the "unconditional" block (line 127):
 elif mode == "contour":
     for item in data:
         json_output = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
         conversation = {
             "messages": [
                 {"role": "user", "content": "Generate B-spline contour slices for a 3D shape in UDF contour token format:"},
                 {"role": "assistant", "content": json_output}
             ]
         }
         formatted_data.append(conversation)
 Update error message (line 189) and docstring (line 30) to include "contour".

 File: src/data/prepare_dataset.py (line 1023) — add "contour" to argparse choices:
 choices=["unconditional", "blueprint", "caption", "panorama", "multi_view", "multi_view2", "multi_view3", "contour"]

 No other changes needed — load_data_by_mode() already falls through to load_json_files() for text-only modes, and tokenize_function() already drops samples exceeding max_length.

 Step 3: Run the pipeline end-to-end

 # Flatten
 python scripts/prepare_contour_data.py \
     --input_dir /home/yhc/cross_sdf/data/thick_structures_slices \
     --output_dir /tmp/contour_flat

 # Tokenize
 python -m src.data.prepare_dataset \
     --json_dir /tmp/contour_flat \
     --output_dir ./data/tokenized/contour_test \
     --model_name "Qwen/Qwen3-8B" \
     --max_length 8192 \
     --mode contour \
     --seed 42 --num_proc 4

 Samples exceeding 8192 tokens will be automatically dropped by tokenize_function() (it returns empty lists which get filtered out).

 Step 4: Verify output

 Use existing scripts/inspect_arrow.py or scripts/stats_arrow_length.py to confirm the Arrow dataset was created and has valid samples.

 Files to Create/Modify

 ┌─────────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
 │              File               │                               Change                                │
 ├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
 │ scripts/prepare_contour_data.py │ NEW — ~30 lines, flatten nested dirs                                │
 ├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
 │ src/data/converters.py          │ Add "contour" elif branch (~15 lines), update docstring + error msg │
 ├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
 │ src/data/prepare_dataset.py     │ Add "contour" to choices list (1 word)                              │
 └─────────────────────────────────┴─────────────────────────────────────────────────────────────────────┘

 Expected Results

 - 17 input files, ~8-9 will survive at max_length=8192 (the simpler objects)
 - Larger objects (brain, air_conditioner, airplane, etc.) will be dropped
 - Arrow dataset created in ./data/tokenized/contour_test/