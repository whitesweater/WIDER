#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import math
import re
import os
import string
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.nn import functional as F
import json

DATA_DIR = os.environ.get(
    "WIDER_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
from accelerate.utils import set_seed
from safetensors.torch import load_file

import numpy as np
from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from src.runtime_device import DEVICE, DEVICE_TYPE, move_model_to_runtime

do_print = True

device = DEVICE
print(device)


def parse_experiment_info(ckpt_dir):
    """Extract the experiment and model names from a standard checkpoint path."""
    parts = os.path.normpath(ckpt_dir).split(os.sep)
    for index, part in enumerate(parts):
        if part.startswith("checkpoint-") and index >= 5:
            return parts[index - 5], parts[index - 4]
    return None, None

import json
from peft import PeftModel, LoraConfig

def save_jsonl_line(filepath, data):
    """
    将一条字典数据追加写入到 JSONL 文件中。

    参数:
        filepath (str): 目标 JSONL 文件路径。
        data (dict): 要写入的数据，必须是可序列化为 JSON 的字典。
    """
    if not isinstance(data, dict):
        raise ValueError("data 必须是一个字典")

    with open(filepath, "a", encoding="utf-8") as f:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + "\n")

def read_json(file_path):
    """
    从指定路径读取JSON文件并返回对应的Python对象。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        return None

def write_json(data, file_path):
    """
    将Python对象写入指定路径的JSON文件中。
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"写入JSON文件时出错: {e}")

def evaluation(model_args, data_args, training_args):
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    else:
        raise NotImplementedError
    # import pdb; pdb.set_trace()
    model = CODI(model_args, training_args, lora_config)
    #if "llama" in model_args.model_name_or_path:
    #    model.codi.resize_token_embeddings(128261)
    # import pdb; pdb.set_trace()
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    
    # new_state_dict = { k.replace("coconut", "codi"): v for k, v in state_dict.items() }
    # torch.save(new_state_dict, "/scratch/prj/inf_multimodal_qa/scratch_tmp/transfer/pytorch_model.bin")
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)
    print(
        "checkpoint load_state_dict strict=False | "
        f"missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
    )
    if missing_keys:
        print("missing sample:", missing_keys[:20])
    if unexpected_keys:
        print("unexpected sample:", unexpected_keys[:20])
    model.codi.tie_weights()
    
    tokenizer_path = model_args.model_name_or_path 
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    device = DEVICE_TYPE
    model, device = move_model_to_runtime(model)
    # import pdb; pdb.set_trace()
    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    question_name = "question"
    answer_name = "answer"
    if "gsm-hard" == data_args.data_name:
        # dataset = load_dataset("juyoung-trl/gsm-hard")
        # test_set = dataset['train']
        # question_name = "instruction"
        # answer_name = "response"
        test_set = read_json(os.path.join(DATA_DIR, 'gsm8k_hard_format.json'))
    elif "multi-arith" == data_args.data_name:
        # dataset = load_dataset("ChilleD/MultiArith")
        # test_set = dataset['test']
        # answer_name = "final_ans"
        test_set = read_json(os.path.join(DATA_DIR, 'multiarith_format.json'))
    elif "svamp" == data_args.data_name:
        # dataset = load_dataset("ChilleD/SVAMP")
        # test_set = concatenate_datasets([dataset["train"], dataset["test"]])
        # question_name = "question_concat"
        # answer_name = "Answer"
        test_set = read_json(os.path.join(DATA_DIR, 'svamp_format.json'))
    elif "commonsense" == data_args.data_name:
        local_path = os.path.join(DATA_DIR, 'commonsense_val_clean.json')
        if os.path.exists(local_path):
            test_set = read_json(local_path)
        else:
            dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
            test_set = dataset['validation']
    elif "csqa2" == data_args.data_name:
        test_set = read_json(os.path.join(DATA_DIR, 'csqa2_dev_ood_clean.json'))
    elif "gsm8k" == data_args.data_name:
        # dataset = load_dataset("gsm8k", "main")
        # test_set = dataset['test']
        test_set = read_json(os.path.join(DATA_DIR, 'gsm_test_clean.json'))
        # import pdb; pdb.set_trace()
        # print()
    elif data_args.data_name in {"asdiv-aug", "asdiv_aug", "asdiv"}:
        test_set = read_json(os.path.join(DATA_DIR, 'asdiv_aug_val_clean.json'))
    elif "aqua" == data_args.data_name:
        test_set = read_json(os.path.join(DATA_DIR, 'aqua_val_clean.json'))
    elif data_args.data_name in {"gsm-mc", "gsm_mc"}:
        test_set = read_json(os.path.join(DATA_DIR, 'gsm_mc_test_ood_clean.json'))
    elif data_args.data_name in {"du", "date_understanding"}:
        test_set = read_json(os.path.join(DATA_DIR, 'du_val_clean.json'))
    elif "musique" == data_args.data_name:
        test_set = read_json(os.path.join(DATA_DIR, 'musique_val_clean.json'))
    elif "hotpotqa" == data_args.data_name:
        test_set = read_json(os.path.join(DATA_DIR, 'hotpotqa_val_clean.json'))
    elif "drop" == data_args.data_name:
        test_set = read_json(os.path.join(DATA_DIR, 'drop_val_clean.json'))
    elif "strategyqa" == data_args.data_name:
        test_set = read_json(os.path.join(DATA_DIR, 'strategyqa_test_clean.json'))
    elif "prontoqa" == data_args.data_name:
        test_set = read_json(os.path.join(DATA_DIR, 'prontoqa_test_clean.json'))
    elif "aime" == data_args.data_name:
        test_set = read_json(os.path.join(DATA_DIR, 'aime_test_clean.json'))
    else:
        raise NotImplementedError

    logging.warning("Formatting inputs...")
    question = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
    answer = []

    is_bool_task = _is_bool_task_name(data_args.data_name)
    is_choice_task = _choice_labels_for_task(data_args.data_name) is not None
    is_text_task = any(t in data_args.data_name for t in ["musique", "hotpotqa", "drop"])

    for example in test_set:
        example = example[answer_name]
        if is_bool_task:
            raw = str(example).strip()
            answer.append(raw)
            continue
        if is_choice_task:
            answer.append(str(example).strip().upper())
            continue
        if is_text_task:
            answer.append(example.strip() if isinstance(example, str) else str(example))
            continue
        if isinstance(example, bool):
            answer.append(example)
            continue
        if example in ["True", "False"]:
            answer.append(True if example == "True" else False)
            continue
        if example in "ABCDEF":
            answer.append(example)
            continue
        if "####" in example:
            ans = example.split('####')[-1]
        else:
            ans = example
        ans = ans.replace(',', '')
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question)/data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")
    
    question_data = []
    # import pdb; pdb.set_trace()
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i*data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )
        
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch.to(device))

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature":0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    raw_decoded_list = []
    ans_pred_list_accu_at_n_passes = []
    attention_map_weights = []
    attention_to_latents_against_len_sum = []
    attention_to_latents_against_len_count = []
    #set_seed(42)
    gating_probs_sums = None
    len_cot = []
    model.eval()
    attn_to_latent_list = []
    if model_args.soft_weight:
        embedding_matrix = model.codi.get_base_model().model.embed_tokens.weight.data.to(model.codi.device)
        vocab_center = embedding_matrix.mean(dim=0)
    
    all_saved_latents = []

    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        with torch.no_grad():
            # encode the question
            past_key_values = None
            outputs = model.codi(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=batch["attention_mask"])
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            if model_args.soft_weight:
                soft_probs = F.softmax(model.codi.lm_head(latent_embd).to(model.codi.device), dim=-1)
                soft_embeds = (soft_probs.squeeze(dim=1).unsqueeze(dim=-1) * embedding_matrix.unsqueeze(dim=0)).sum(dim=0)

                soft_probs_expanded = soft_probs.squeeze(dim=1)
                soft_embeds = (soft_probs_expanded.unsqueeze(dim=2) * embedding_matrix).sum(dim=1)  # Shape: [128, 2048]
                if training_args.use_prj:
                    soft_embeds = model.prj(soft_embeds)
                latent_embd = latent_embd + model_args.soft_weight * soft_embeds
            
            all_latents_for_sample = (
                [latent_embd.detach().cpu().float().squeeze(1)]
                if training_args.save_latents
                else None
            )

            inf_latent_iterations = training_args.inf_latent_iterations
            for i in range(inf_latent_iterations):
                # decode the latent embeddings
                outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                
                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)
                
                if model_args.soft_weight:
                    soft_probs = F.softmax(model.codi.lm_head(latent_embd).to(model.codi.device), dim=-1)
                    soft_embeds = (soft_probs.squeeze(dim=1).unsqueeze(dim=-1) * embedding_matrix.unsqueeze(dim=0)).sum(dim=0)
                    
                    soft_probs_expanded = soft_probs.squeeze(dim=1)
                    soft_embeds = (soft_probs_expanded.unsqueeze(dim=2) * embedding_matrix).sum(dim=1)  # Shape: [128, 2048]
                    # import pdb; pdb.set_trace()
                    if training_args.use_prj:
                        soft_embeds = model.prj(soft_embeds)

                    latent_embd = latent_embd + model_args.soft_weight * soft_embeds

                if all_latents_for_sample is not None:
                    all_latents_for_sample.append(
                        latent_embd.detach().cpu().float().squeeze(1)
                    )

            if training_args.save_latents:
                # Stack latents: shape (batch, num_iterations+1, hidden_dim)
                assert all_latents_for_sample is not None
                latent_matrix = torch.stack(all_latents_for_sample, dim=1)
                for b in range(batch_size):
                    all_saved_latents.append(latent_matrix[b].numpy().tolist())

            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device=device)).unsqueeze(0).to(device)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device=device)).unsqueeze(0).to(device)
            
            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)

            output = eot_emb
            
            seq_len = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)  # Track EOS for each sequence
            pred_tokens = [[] for _ in range(batch_size)]
            for i in range(gen_kwargs["max_new_tokens"]):
                seq_len += 1
                out = model.codi(
                        inputs_embeds=output,
                        output_hidden_states=False,
                        attention_mask=None,
                        use_cache=True,
                        output_attentions=False,
                        past_key_values=past_key_values
                    )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

                # implement the sampling process
                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False

                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # Handle EOS for each sequence
                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                # Break if all sequences have finished
                if finished.all():
                    break

                #output = model.codi.get_base_model().transformer.wte(next_token_ids).unsqueeze(1).to(device)
                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            for mini_step, pred_token in enumerate(pred_tokens):
                # import pdb; pdb.set_trace()
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                # os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                # save_jsonl_line(output_dir, {'list': tokenizer.encode(cot_output+answer_output, add_special_tokens=False), 'length': len(tokenizer.encode(cot_output+answer_output, add_special_tokens=False))})
                # import pdb; pdb.set_trace()
                # Extract the numbers in sentences 
                if do_print:
                    print(f"Question {step*data_args.batch_size+mini_step} Starts...")
                    print(f"Q: {question[step*data_args.batch_size+mini_step]}")
                    print(decoded_pred)
                    print(f"Question {step*data_args.batch_size+mini_step} Ends")
                    print(f"Prediction={extract_answer_number(decoded_pred)}; Groundtruth={answer[step*data_args.batch_size+mini_step]}")
                    print("")
                ans_pred_list.append(extract_answer_number(decoded_pred))
                raw_decoded_list.append(decoded_pred)
    ckpt_name = os.path.basename(model_args.ckpt_dir.rstrip('/'))
    expt_name, model_short = parse_experiment_info(model_args.ckpt_dir)
    if expt_name is None:
        expt_name = training_args.expt_name
    if model_short is None:
        model_short = model_args.model_name_or_path.split('/')[-1]

    result_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    latent_dir = os.path.join(result_base, expt_name, model_short, ckpt_name, data_args.data_name, "run_0")
    os.makedirs(latent_dir, exist_ok=True)

    ans_file = os.path.join(latent_dir, "answers.json")
    write_json({"ans": ans_pred_list}, ans_file)
    print(f"Saved answers to {ans_file}")

    debug_output_path = os.path.join(latent_dir, "debug.jsonl")
    with open(debug_output_path, "w", encoding="utf-8") as dbg_f:
        for idx in range(len(ans_pred_list)):
            dbg_record = {
                "index": idx,
                "checkpoint": ckpt_name,
                "question": question[idx] if idx < len(question) else "",
                "raw_output": raw_decoded_list[idx] if idx < len(raw_decoded_list) else "",
                "ground_truth": str(answer[idx]),
                "prediction": str(ans_pred_list[idx]),
                "gt_type": type(answer[idx]).__name__,
                "pred_type": type(ans_pred_list[idx]).__name__,
            }
            dbg_f.write(json.dumps(dbg_record, ensure_ascii=False) + "\n")
    print(f"Debug output saved to {debug_output_path} ({len(ans_pred_list)} samples)")

    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"adapter: {model_args.adapter_name_or_path} | {data_args.data_name} test accuracy: {100*accuracy:.2f}% | ")
    print(f"average length of COT: {sum(len_cot)/len(len_cot)}")
    # import pdb; pdb.set_trace()
    if model_args.save_ablation:
        save_jsonl_line(os.path.join(latent_dir, "ablation.jsonl"), {'model_name': '-'.join(model_args.ckpt_dir.split('/')[5:]), 'data_name': data_args.data_name, 'soft_weight': model_args.soft_weight, 'acc.': accuracy})

    predictions_file = os.path.join(latent_dir, "predictions.jsonl")
    with open(predictions_file, "w", encoding="utf-8") as f:
        for idx in range(len(ans_pred_list)):
            record = {
                "index": idx,
                "checkpoint": ckpt_name,
                "question": question[idx] if idx < len(question) else "",
                "raw_output": raw_decoded_list[idx] if idx < len(raw_decoded_list) else "",
                "prediction": ans_pred_list[idx],
                "ground_truth": answer[idx] if idx < len(answer) else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(ans_pred_list)} predictions to {predictions_file}")

    if training_args.save_latents:
        latent_file = os.path.join(latent_dir, "latents.json")
        with open(latent_file, "w") as f:
            json.dump({"latents": all_saved_latents}, f)
        print(f"Saved {len(all_saved_latents)} latent matrices to {latent_file}")
    else:
        print("Skipped latent-state serialization (--save_latents False)")

    metrics_file = os.path.join(latent_dir, "metrics.json")
    avg_cot = sum(len_cot) / len(len_cot) if len_cot else 0
    with open(metrics_file, "w") as f:
        json.dump({"accuracy": accuracy, "avg_cot_length": avg_cot}, f)

    return 100*accuracy

def _extract_choice(sentence: str, choices: str = "ABCDE") -> str:
    target = sentence.split("The answer is:")[-1] if "The answer is:" in sentence else sentence
    target = target.strip()
    if not target:
        return "?"
    match = re.search(rf"\b([{re.escape(choices)}])\b", target.upper())
    if match:
        return match.group(1)
    first = target[0].upper()
    return first if first in choices else "?"


def _extract_commonsense_choice(sentence: str) -> str:
    return _extract_choice(sentence, "ABCDE")


def _is_bool_task_name(data_name: str) -> bool:
    normalized = str(data_name).strip().lower()
    return normalized in {"strategyqa", "prontoqa", "csqa2"}


def _choice_labels_for_task(data_name: str) -> str | None:
    normalized = str(data_name).strip().lower()
    if normalized in {"gsm-mc", "gsm_mc"}:
        return "ABCD"
    if "commonsense" in normalized or "aqua" in normalized:
        return "ABCDE"
    if normalized in {"du", "date_understanding"}:
        return "ABCDEF"
    return None


def extract_answer_number(sentence: str) -> float:
    data_name = getattr(globals().get("data_args", None), "data_name", "")
    choice_labels = _choice_labels_for_task(data_name)
    if choice_labels is not None:
        return _extract_choice(sentence, choice_labels)
    if "The answer is:" in sentence:
        raw = sentence.split("The answer is:")[-1].strip().rstrip(".")
        if not raw:
            return "?"
        # For non-text tasks, try to convert to float for consistent comparison
        cleaned = raw.replace(',', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            return raw
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        if _is_bool_task_name(data_name):
            lower = sentence.lower()
            if "true" in lower:
                return "True"
            elif "false" in lower:
                return "False"
            return "?"
        return float('inf')

    pred_answer = float(pred[-1])

    return pred_answer


def _normalize_text_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', value)

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in value if ch not in exclude)

    def white_space_fix(value: str) -> str:
        return ' '.join(value.split())

    return white_space_fix(remove_articles(remove_punc(text.lower().strip())))


def _is_choice_label(value, choices: str = "ABCDEF") -> bool:
    return isinstance(value, str) and value.strip().upper() in choices and len(value.strip()) == 1


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        elif _is_choice_label(g) and _is_choice_label(p):
            if str(p).strip().upper() == str(g).strip().upper():
                acc += 1
        elif isinstance(g, str) and isinstance(p, str):
            normalized_gold = _normalize_text_answer(g)
            normalized_pred = _normalize_text_answer(p)
            if normalized_gold and normalized_pred and normalized_gold == normalized_pred:
                acc += 1
        elif isinstance(g, str) or isinstance(p, str):
            # Try numeric comparison first to handle float vs string mismatch
            try:
                if float(str(p).replace(',', '').strip()) == float(str(g).replace(',', '').strip()):
                    acc += 1
                    continue
            except (ValueError, TypeError):
                pass
            if str(p).lower().strip() == str(g).lower().strip():
                acc += 1
        else:
            if p == g:
                acc += 1

    return acc / len(gold)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accu_list = []
    for i in range(training_args.inf_num_iterations):
        accu = evaluation(model_args, data_args, training_args)
        accu_list.append(accu)
    print(f"Average accuracy over {training_args.inf_num_iterations} sampling: {sum(accu_list)/len(accu_list)}")
