    # CRITICAL FIX: ShadowOps GRPO Trainable Parameters Issue

    ## Summary

    **Problem**: The GRPO training pipeline was frozen with 0 trainable parameters, causing loss=0.0 and grad_norm=0.0 throughout training despite proper LoRA configuration.

    **Root Cause**: Missing `prepare_model_for_int4_training()` call before applying PEFT adapters, combined with insufficient validation to catch the issue early.

    **Solution Applied**: 
    1. ✅ Added explicit `prepare_model_for_int4_training()` before `get_peft_model()`
    2. ✅ Added comprehensive diagnostic logging
    3. ✅ Added validation to fail early if trainable_params == 0
    4. ✅ Created helper utilities for quick testing

    ---

    ## Files Modified

    ### 1. `backend-ml/training/train_qwen3_grpo.py` (Main Fix)

    **Changes:**
    - **Line 988-1013**: Added `print_trainable_modules()` function to list all trainable modules for diagnostics
    - **Line 1260-1272**: Added `prepare_model_for_int4_training()` call with error handling
    - **Line 1274-1299**: Reorganized LoRA application with detailed logging
    - **Line 1391-1406**: Added trainable parameter validation checkpoint with early exit if count=0
    - **Line 1409-1435**: Added module analysis and improvement reporting after training

    **Key Pattern:**
    ```python
    # Step 1: Load model with 4-bit quantization
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # Step 2: Prepare for int4 training (CRITICAL FIX)
    from peft import prepare_model_for_int4_training
    model = prepare_model_for_int4_training(model)

    # Step 3: Apply LoRA adapters
    model = FastModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        bias="none",
    )

    # Step 4: Validate that adapters were created (NEW VALIDATION)
    trainable_params, total_params = count_trainable_parameters(model)
    if trainable_params == 0:
        print("[ERROR] No trainable parameters found!")
        return None  # Fail fast

    # Step 5: Training proceeds with non-zero gradients
    FastModel.for_training(model)
    trainer.train()  # Now works correctly
    ```

    ---

    ## New Diagnostic Tools

    ### 2. `backend-ml/training/test_lora_setup.py` (NEW)

    **Purpose**: Quick 2-minute diagnostic test to verify LoRA configuration

    **Checks**:
    1. Dependencies installed (torch, transformers, unsloth, peft)
    2. GPU availability
    3. Model loads successfully with 4-bit quantization
    4. **Trainable parameters > 0 after PEFT** ← Key indicator
    5. Adapter modules created correctly
    6. Gradient flow works during backward pass

    **Usage**:
    ```bash
    cd backend-ml
    python training/test_lora_setup.py --model 1.7b
    ```

    **Expected Output if Fix Works**:
    ```
    [6] Analyzing model after PEFT...
    Total params: 1,650,000,000
    Trainable: 4,194,816 (0.25%)
    New adapter params: 4,194,816
    ✓ PASS: LoRA created 4,194,816 trainable parameters

    [8] Testing gradient flow...
    ✓ Gradient flow OK: 56 parameters have non-zero gradients

    ======================================================================
    RESULT: LoRA SETUP IS CORRECT ✓
    ======================================================================
    ```

    ### 3. `backend-ml/training/evaluate_policy_comparison.py` (NEW)

    **Purpose**: Comprehensive policy evaluation and comparison

    **Compares**:
    - Random (baseline - uniform 4-action)
    - Heuristic (baseline - rule-based, no quarantine)
    - QuarantineAware (reference - reward-optimized heuristic)
    - Oracle (ceiling - theoretical upper bound)
    - Trained (your GRPO model, if checkpoint provided)

    **Metrics Per Policy**:
    - Mean reward ± std deviation
    - Accuracy (threat-level correct classification)
    - False positive rate (benign blocked)
    - Action distribution (ALLOW/BLOCK/FORK/QUARANTINE %)

    **Usage**:
    ```bash
    # Baseline comparison only (15 min, no GPU required)
    python training/evaluate_policy_comparison.py --num-episodes 50

    # With trained model (15 min with GPU)
    python training/evaluate_policy_comparison.py \
    --model-path ./shadowops_qwen3_1p7b_model \
    --num-episodes 100 \
    --save-csv comparison_results.csv \
    --save-json comparison_results.json
    ```

    ---

    ## Validation Checklist

    ### Phase 1: Quick Diagnostic (2 min)
    ```bash
    python training/test_lora_setup.py --model 1.7b
    ```
    - [ ] Shows "✓ PASS: LoRA created X,XXX,XXX trainable parameters"
    - [ ] Shows "✓ Gradient flow OK"

    ### Phase 2: Short Training Test (10 min, requires GPU)
    ```bash
    python training/train_qwen3_grpo.py --model 1.7b --max-steps 20
    ```
    - [ ] Prints "Trainable: X,XXX,XXX / 1,650,000,000 (0.25%)"
    - [ ] Loss decreases (not stuck at 0.0)
    - [ ] grad_norm is non-zero
    - [ ] Saves checkpoint successfully

    ### Phase 3: Baseline Comparison (15 min, no GPU)
    ```bash
    python training/evaluate_policy_comparison.py --num-episodes 50 --save-csv results.csv
    ```
    - [ ] QuarantineAware reward > Heuristic reward
    - [ ] All action types (ALLOW/BLOCK/FORK/QUARANTINE) appear

    ### Phase 4: Full Training (30-60 min, requires GPU)
    ```bash
    python training/train_qwen3_grpo.py --model 1.7b
    ```
    - [ ] Trainable parameters printed correctly
    - [ ] Loss non-zero and decreasing
    - [ ] Post-train validation metrics > pre-train metrics
    - [ ] Checkpoint saved to `./shadowops_qwen3_1p7b_model`

    ### Phase 5: Full Comparison with Trained Model (15 min, requires GPU)
    ```bash
    python training/evaluate_policy_comparison.py \
    --model-path ./shadowops_qwen3_1p7b_model \
    --num-episodes 100 \
    --save-csv final_results.csv
    ```
    - [ ] Trained policy reward >= QuarantineAware reward
    - [ ] Valid action rate = 100%
    - [ ] False positive rate reasonable

    ---

    ## Technical Details

    ### Expected Trainable Parameters
    - **Model**: Qwen3-1.7B (1.65B total params)
    - **LoRA Configuration**:
    - r=8 (adapter rank)
    - lora_alpha=16 (scaling)
    - target_modules: 7 (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
    - Each module: ~12 layers × 256 hidden dim × 8 rank ≈ ~100k params
    - **Expected Trainable**: ~4-5M parameters (0.25% of model)
    - **Base Model Params**: All frozen (0 trainable)

    ### Training Configuration
    ```
    batch_size = 1
    grad_accum = 2
    learning_rate = 1e-5
    max_steps = 300
    warmup_ratio = 0.05
    optimizer = paged_adamw_8bit
    ```

    ### GRPO Configuration
    ```
    num_generations = 2              # Samples per prompt
    temperature = 1.0               # Wide sampling
    top_p = 1.0                      # No truncation
    reward_fn = imitation + task     # Behavior + return-to-go
    ```

    ---

    ## What NOT to Change

    ✅ **PRESERVED**:
    - 4-action design (ALLOW/BLOCK/FORK/QUARANTINE)
    - UniversalShadowEnv and quarantine state machine
    - QuarantineAwarePolicy as teacher
    - Environment reward structure
    - LoRA target modules for Qwen3
    - CLI interface (backward compatible)

    ❌ **DO NOT**:
    - Remove or rename actions
    - Change environment observation/action space
    - Modify base reward function
    - Disable gradient checkpointing
    - Use different optimizer (paged_adamw_8bit is critical for VRAM)

    ---

    ## Troubleshooting

    | Issue | Check | Fix |
    |-------|-------|-----|
    | "No trainable parameters" after PEFT | Run `test_lora_setup.py` | Ensure `prepare_model_for_int4_training()` is called |
    | Loss stays at 0.0 | Check grad_norm in trainer log | LoRA not properly enabled; check model.peft_config |
    | Out of memory | Reduce batch_size or max_seq_len | Use 1.7b model (smallest) and batch_size=1 |
    | Slow training | CPU training detected | Ensure CUDA available: `python -c "import torch; print(torch.cuda.is_available())"` |
    | Invalid action tokens | Check tokenizer format | Use `format_prompt_for_model()` with Qwen's apply_chat_template |

    ---

    ## Success Criteria

    **Minimum Requirements**:
    1. Trainable parameters > 0 (should be ~4.2M)
    2. Loss non-zero and decreasing during training
    3. grad_norm non-zero (confirms gradient flow)
    4. Pre-train vs post-train validation metrics differ

    **Target Performance**:
    1. Trained GRPO reward ≥ QuarantineAware reward
    2. Valid action rate = 100%
    3. False positive rate ≤ Heuristic FPR
    4. All 4 actions represented in output

    ---

    ## Next Steps

    1. **Run diagnostic test** (2 min): `python training/test_lora_setup.py --model 1.7b`
    2. **Run short training** (10 min): `python training/train_qwen3_grpo.py --model 1.7b --max-steps 20`
    3. **Run full training** (30-60 min): `python training/train_qwen3_grpo.py --model 1.7b`
    4. **Compare results** (15 min): `python training/evaluate_policy_comparison.py --model-path ./shadowops_qwen3_1p7b_model`

    **Total Time**: ~2-1.5 hours to full validation

    ---

    ## Questions?

    If `test_lora_setup.py` fails at any step:
    1. Check dependency versions: `pip show torch transformers unsloth peft`
    2. Verify CUDA: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
    3. Check exact error message and search for similar issues in:
    - Unsloth GitHub: https://github.com/unslothai/unsloth
    - PEFT GitHub: https://github.com/huggingface/peft
    - TRL GitHub: https://github.com/huggingface/trl
