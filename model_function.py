import os
import time
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Function to load the model for text generation
def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        low_cpu_mem_usage=True,
        device_map = 'auto',
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    return model

# Run inference for a specific task
def main(
    model,
    tokenizer,
    BASE_PROMPT,
    task_instruction, #this needs to be commented out because we are using instruction in file
    dataset,
    csv_file_path,
    custom_instruct = False,
    sample_size=4,
    max_new_tokens=100,
    seed=42,
    do_sample=True,
    min_length=None,
    use_cache=True,
    top_p=1.0,
    temperature=0.5,
    top_k=5,
    repetition_penalty=1.2,
    length_penalty=1,
    **kwargs
):
    model.eval()
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['ID','Instruction', 'Input Text', 'Response', 'Log-Likelihood', 'Targets', 'Task', 'Langs'])

        for i, item in enumerate(dataset):
            if i >= sample_size:
                break

            if custom_instruct == False:
              instruction = item['instruction']
            else:
              instruction = task_instruction
            input_text = item['inputs']
            labels = item['targets']
            langs = item['langs']
            try:
              task = item['task']
            except:
              task = 'xnli'
            identity = item['ID']

            user_prompt = BASE_PROMPT.format(f"{instruction}\n{input_text}")
            batch = tokenizer(user_prompt, return_tensors="pt")
            batch = {k: v.to(model.device) for k, v in batch.items()}

            start = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    min_length=min_length,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    **kwargs
                )

            # e2e_inference_time = (time.perf_counter() - start) * 1000
            # print(f"Inference time: {e2e_inference_time} ms")

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(user_prompt):]

            if task !="mmt":
              with torch.no_grad():
                  logits = model(**batch).logits
                  log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                  generated_tokens = outputs[:, len(batch['input_ids'][0]):]
                  log_likelihood = log_probs.gather(2, generated_tokens.unsqueeze(-1)).squeeze(-1).sum().item()
            else:
              log_likelihood = []
            writer.writerow([identity, instruction, input_text, output_text, log_likelihood, labels, task, langs])  # Save ground truth
