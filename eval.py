from unsloth import FastLanguageModel


def test(model, tokenizer, test_tasks):
    import lm_eval.models.huggingface
    import lm_eval
    ac_lm_eval = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
    results = lm_eval.simple_evaluate(
        ac_lm_eval,
        tasks=test_tasks,
        log_samples=False,
        limit=100,
    )
    
    return results

test_tasks = ["gsm8k", "ifeval", "mmlu"]

def main(model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = "bfloat16",
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    results = test(model, tokenizer, test_tasks)
    print("*" * 40)
    print("Evaluation Results for Model:", model_path)
    print(results)
    print("*" * 40)

if __name__ == "__main__":
    model_names = [
        "/home/xiangyu/Documents/ID2223Lab2/lora_models/unsloth-Llama-3.2-1B-Instruct-lora-r8-FineTome",
        "/home/xiangyu/Documents/ID2223Lab2/lora_models/unsloth-Llama-3.2-1B-Instruct-lora-r16-FineTome",
        "/home/xiangyu/Documents/ID2223Lab2/lora_models/unsloth-Llama-3.2-1B-Instruct-lora-r32-FineTome",
        "/home/xiangyu/Documents/ID2223Lab2/lora_models/unsloth-Llama-3.2-3B-Instruct-lora-r8-FineTome",
        "/home/xiangyu/Documents/ID2223Lab2/lora_models/unsloth-Llama-3.2-3B-Instruct-lora-r16-FineTome",
        "/home/xiangyu/Documents/ID2223Lab2/lora_models/unsloth-Llama-3.2-3B-Instruct-lora-r32-FineTome",
        "/home/xiangyu/Documents/ID2223Lab2/lora_models/unsloth-Qwen2.5-3B-Instruct-lora-r8-FineTome",
        "/home/xiangyu/Documents/ID2223Lab2/lora_models/unsloth-Qwen2.5-3B-Instruct-lora-r16-FineTome",
        "/home/xiangyu/Documents/ID2223Lab2/lora_models/unsloth-Qwen2.5-3B-Instruct-lora-r32-FineTome",
    ]
    for model_name in model_names:
        main(model_name)