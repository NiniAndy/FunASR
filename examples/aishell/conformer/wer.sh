    echo "Computing WER ..."
    # 推理的结果
    inference_dir=/ssd/zhuang/code/FunASR/examples/industrial_data_pretraining/llm_asr/exp/llm_asr_whisper_qwen_aishell1_exp3.bac
    # ground truth
    data_dir=/ssd/zhuang/code/FunASR/examples/aishell/DATA/data/test

    python utils/postprocess_text_zh.py ${inference_dir}/1best_recog/text ${inference_dir}/1best_recog/text.proc
    python utils/postprocess_text_zh.py  ${data_dir}/text ${inference_dir}/1best_recog/text.ref
    python utils/compute_wer.py ${inference_dir}/1best_recog/text.ref ${inference_dir}/1best_recog/text.proc ${inference_dir}/1best_recog/text.cer
    tail -n 3 ${inference_dir}/1best_recog/text.cer