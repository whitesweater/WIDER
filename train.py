# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import copy
import logging
import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import re
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import json
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from safetensors.torch import load_file
from tqdm import tqdm
from math import ceil
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from functools import partial
from tqdm import tqdm
from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    freeze_model
)
from src.dataset_cache import build_dataset_cache_path, load_dataset_cache, save_dataset_cache
from src.dataset_cache import should_save_dataset_cache
from src.runtime_device import DEVICE
import json


def _to_scalar(x):
    """Convert Tensor/number/None to python float (mean-reduced if needed)."""
    import torch
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        # detach，转到 float，若多元素则取 mean，再 item()
        return x.detach().float().mean().item()
    # 已经是数字的情况
    return float(x)
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
IGNORE_INDEX = -100

device = DEVICE
print(device)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch):
        # Extract the global step from the optimizer
        step = self.state.global_step

        # Get total training steps
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)

        effective_batch_size = batch_size * self.args.world_size * gradient_accumulation_steps
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        # Call the model's forward method
        outputs = model(**inputs)
        loss = outputs["loss"]
        loss_for_log = loss.detach() if isinstance(loss, torch.Tensor) else loss
        if step % self.args.logging_steps == 0:
            logs = {
                "loss": _to_scalar(loss_for_log),
                "ce_loss": _to_scalar(outputs.get("ce_loss")),
                "distill_loss": _to_scalar(outputs.get("distill_loss")),
                "ref_ce_loss": _to_scalar(outputs.get("ref_ce_loss")),
                "spectral_loss": _to_scalar(outputs.get("spectral_loss")),
            }
            if not hasattr(self, "is_global_zero") or self.is_global_zero:
                self.log(logs)
        #"ce_loss": ce_loss_total, "mse_loss": mse_loss_total, "ref_ce_loss": ref_ce_loss
        # if step % self.args.logging_steps == 0:
        #     self.log({"loss": loss.item(), "ce_loss": outputs["ce_loss"], "distill_loss": outputs["distill_loss"], "ref_ce_loss": outputs["ref_ce_loss"],})

        return loss

    def log(self, logs, start_time=None):
        if self.state.global_step is not None:
            for k, v in logs.items():
                super().log({k: v})

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=256,#training_args.model_max_length,
            truncation=True,
            return_attention_mask=False
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = [sentence]
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ##########################
    #       Peft Model       #
    ##########################
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2", "gsm-cot"]):
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

    # import pdb; pdb.set_trace()
    model = CODI(model_args, training_args, lora_config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    def get_answer_token_position(tokens, answer_prompts, tokenizer):
        #answer_prompt = torch.tensor([464, 3280, 318, 25])
        # import pdb; pdb.set_trace()
        try:
            match_indices = (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]).all(dim=1).nonzero(as_tuple=True)[0].item()
            answer_token_id = match_indices + len(answer_prompts[0])
            return answer_token_id
        except Exception:
            breakpoint()
        
    # def get_steps_

    def preprocess(
        sources: Sequence[str], 
        targets: Sequence[str], 
        answers: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer, 
        bot_id: int,
        eot_id: int,
    ) -> Dict:
        print("Tokenizing inputs... This may take some time...")
        sources_id = _tokenize_fn(sources, tokenizer)["input_ids"]
        cot_id = _tokenize_fn(targets, tokenizer)["input_ids"]
        answers_id = _tokenize_fn(answers, tokenizer)["input_ids"]

        # add eos token to accomodate pretrained model's format
        if not training_args.remove_eos:
            sources_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
            cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
        answers_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in answers_id]

        if cot_id[0][0] == tokenizer.bos_token_id:
            cot_id = [x[1:] for x in cot_id]
            answers_id = [x[1:] for x in answers_id]

        ref_input_ids = [torch.cat([x, y, z]).to(torch.long) for x, y, z in zip(sources_id, cot_id, answers_id)]
        ref_labels = []
        for x, y in zip(ref_input_ids, sources_id):
            z = x.clone()
            z[:len(y)] = -100
            ref_labels.append(z)
        
        # add eot to source
        sources_id = [torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long) for x in sources_id]
        # add eot and eos
        if training_args.remove_eos:
            answers_id = [torch.tensor([eot_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]
        else:
            answers_id = [torch.tensor([eot_id, tokenizer.eos_token_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]

        answer_prompts = [torch.tensor(tokenizer.encode("The answer is:")), torch.tensor(tokenizer.encode("The next step result is:"))]
        if answer_prompts[0][0] == tokenizer.bos_token_id: # remove the bos
            answer_prompts[0] = answer_prompts[0][1:]
            answer_prompts[1] = answer_prompts[1][1:]
        # import pdb; pdb.set_trace()
        ref_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for i, x in enumerate(ref_input_ids)]
        model_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for x in answers_id]

        ref_eos_position = [len(x)-1 for x in ref_input_ids]
        model_eos_position = [len(x)-1 for x in answers_id]
        return dict(encoder_input_ids=sources_id, decoder_input_ids=answers_id, ref_input_ids=ref_input_ids, labels=answers_id, \
                    ref_answer_position=ref_answer_position, model_answer_position=model_answer_position, \
                        ref_eos_position=ref_eos_position, model_eos_position=model_eos_position, ref_labels=ref_labels)


    class SupervisedDataset(Dataset):
        QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
        QUESTION_DA_PROMPT = "\nAnswer the above question. Answer the final number directly in one number.\n"
        def __init__(self, data_name, raw_data, tokenizer, bot, eot):
            super(SupervisedDataset, self).__init__()
            logging.warning("Formatting inputs...")
            
            self.data_name = data_name
            questions, cots, answers = [], [], []
            num_ops_list = []
            operators = ["+", "-", "*", "/"]

            token_nums = []

            _data_dir = os.environ.get(
                "WIDER_DATA_DIR",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
            )
            _data_file_map = {
                "musique": "musique_train_clean.json",
                "hotpotqa": "hotpotqa_train_clean.json",
                "drop": "drop_train_clean.json",
                "strategy": "strategyqa_train_clean.json",
                "prontoqa": "prontoqa_train_clean.json",
                "commonsense": "commonsense_train_clean.json",
                "asdiv": "asdiv_aug_train_clean.json",
                "aqua": "aqua_train_clean.json",
                "gsm8k": "gsm_train_clean.json",
                "icot": "gsm_train_clean.json",
            }
            _data_file = next((v for k, v in _data_file_map.items() if k in self.data_name), "gsm_train_clean.json")
            raw_data_path = os.path.join(_data_dir, _data_file)
            dataset_cache_dir = os.environ.get("WIDER_DATASET_CACHE_DIR")
            if dataset_cache_dir is None and training_args.cache_dir:
                dataset_cache_dir = os.path.join(training_args.cache_dir, "dataset_cache")
            cache_path = build_dataset_cache_path(
                data_name=self.data_name,
                raw_data_path=raw_data_path,
                cache_dir=dataset_cache_dir,
                tokenizer_name=getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__),
                tokenizer_vocab_size=len(tokenizer),
                tokenizer_pad_token_id=tokenizer.pad_token_id,
                tokenizer_bos_token_id=tokenizer.bos_token_id,
                tokenizer_eos_token_id=tokenizer.eos_token_id,
                bot_id=bot,
                eot_id=eot,
                settings={
                    "include_last_cot": training_args.include_last_cot,
                    "max_token_num": training_args.max_token_num,
                    "num_latent": training_args.num_latent,
                    "remove_eos": training_args.remove_eos,
                    "exp_mode": training_args.exp_mode,
                    "exp_data_num": training_args.exp_data_num,
                },
            )
            cached_data = load_dataset_cache(cache_path)

            if cached_data is not None:
                self.data_dict = cached_data["data_dict"]
                self.keys = cached_data["keys"]
                logging.warning(f"Loaded dataset cache from {cache_path} ({len(self)} samples)")
                return

            raw_data = read_json(raw_data_path)
            for num_iter, example in tqdm(enumerate(raw_data)):
                if 'cot' not in example: 
                    example['cot'] = example['steps']
                    example['cot'] = ' '.join(example['cot'])
                if training_args.exp_mode and num_iter > training_args.exp_data_num:
                    break
                question = f"{example['question']}"
                if "icot" in self.data_name and "full" in self.data_name:
                    if example["answer"] is None or example["response"] is None:
                        continue
                    
                    token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                    if token_num > training_args.max_token_num:
                        continue
 
                    cot = f"{example['cot']}".split(". ")
                    if not (training_args.include_last_cot):
                        cot = cot[:-1]
                    answer = f"The answer is: {example['answer'].split(' ')[-1]}"
                    answer = answer.replace("####", "")
                    questions.append(question)
                    cots.append(". ".join(cot)+".\n")
                    answers.append(answer)
                elif "icot" in self.data_name:
                    token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                    if token_num > training_args.max_token_num:
                        continue
 
                    cot_list = []
                    cot = f"{example['cot']}".split(" ")
                    if not training_args.include_last_cot:
                        cot = cot[:-1]
                    
                    len_cot = len(cot) 
                    for i in range(training_args.num_latent):
                        cot_list.append(" ".join(cot[:max(0, len_cot-i)]))
                    answer = example['answer'].split(' ')[-1]
                    
                    if not answer[0].isdigit():
                        continue

                    answer = f"The answer is: {answer}" 
                    answer = answer.replace("####", "")
                    questions.append(question)
                    cots.append(" ".join(cot))
                    answers.append(answer)
                elif "commonsense" in self.data_name or "strategy" in self.data_name:
                    question = example['question'].strip() + '\n'
                    cot = example['cot'].strip() + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"
                    
                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num: 
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                elif "aqua" in self.data_name:
                    question = example['question'].strip() + '\n'
                    cot = example['cot'].strip() + "\n"
                    answer = f"The answer is: {str(example['answer']).strip().upper()}"

                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num:
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                elif "prontoqa" in self.data_name:
                    question = example['question'].strip() + '\n'
                    cot = '\n'.join(example['steps'][:-1]) + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"
                    
                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num: 
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                elif "musique" in self.data_name or "hotpotqa" in self.data_name or "drop" in self.data_name:
                    question = example['question'].strip() + '\n'
                    cot = example['cot'].strip() + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"

                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num:
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                elif "gsm" in self.data_name or "aime" in self.data_name or "asdiv" in self.data_name:
                    question = example['question'].strip() + '\n'
                    cot = example['cot'].strip() + "\n"
                    answer_raw = str(example['answer']).strip().replace("####", "").strip()
                    answer = f"The answer is: {answer_raw}"

                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num:
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                else:
                    raise NotImplementedError(f"Unknown data_name: {self.data_name}")
            if training_args.exp_mode:
                questions = questions[:training_args.exp_data_num]
                cots = cots[:training_args.exp_data_num]
                answers = answers[:training_args.exp_data_num]
            
            print(f"{len(cots)} data in total...")
            logging.warning("Tokenizing inputs... This may take some time...")

            self.data_dict = preprocess(questions, cots, answers, tokenizer, bot, eot)
            self.keys = list(self.data_dict.keys())
            if should_save_dataset_cache():
                try:
                    save_dataset_cache(cache_path, self.data_dict, self.keys)
                    logging.warning(f"Saved dataset cache to {cache_path}")
                except Exception as exc:
                    logging.warning(f"Failed to save dataset cache at {cache_path}: {exc}")
            else:
                logging.warning(f"Skipping dataset cache save on non-primary rank for {cache_path}")


        def __len__(self):
            return len(self.data_dict["encoder_input_ids"])

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {key: self.data_dict[key][i] for key in self.keys}

    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""
        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            encoder_input_ids, decoder_input_ids, ref_input_ids, labels, ref_answer_position, model_answer_position, ref_labels= \
                tuple([instance[key] for instance in instances] for key in ("encoder_input_ids", "decoder_input_ids", "ref_input_ids", "labels", "ref_answer_position", "model_answer_position", "ref_labels"))
        
            # pad left
            reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
            encoder_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(1)
            
            # pad
            ref_input_ids = torch.nn.utils.rnn.pad_sequence(ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            ref_labels = torch.nn.utils.rnn.pad_sequence(ref_labels, batch_first=True, padding_value=IGNORE_INDEX) 

            decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
          
            return dict(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                ref_input_ids=ref_input_ids,
                labels=labels,
                encoder_attention_mask=encoder_input_ids.ne(self.tokenizer.pad_token_id),
                ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),
                model_answer_position=torch.tensor(model_answer_position, dtype=torch.long),
                ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
                ref_labels=ref_labels,
            )

    def make_supervised_data_module(tokenizer, data_args) -> Dict:
        """Make dataset and collator for supervised fine-tuning.
        All datasets are loaded from local data/ JSON files inside SupervisedDataset.
        """
        logging.warning("Loading Data")
        _supported = ["icot", "strategy", "commonsense", "prontoqa", "musique",
                       "hotpotqa", "drop", "gsm8k", "gsm", "aime", "asdiv",
                       "aqua"]
        if not any(s in data_args.data_name for s in _supported):
            raise NotImplementedError(f"Dataset {data_args.data_name} is not supported. "
                                      f"Supported keywords: {_supported}")
        train_dataset = SupervisedDataset(
            data_name=data_args.data_name, raw_data=None,
            tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = CustomTrainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
    trainer.train()

    # to avoid the error of saving the model
    #if "llama" in model_args.model_name_or_path:
    #    trainer.model.codi.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.model.lm_head.weight.clone())
    #if "gpt2" in model_args.model_name_or_path:
    #    trainer.model.codi.transformer.wte.weight = torch.nn.Parameter(model.codi.lm_head.weight.clone())
    #if "qwen" in model_args.model_name_or_path.lower():
    #    trainer.model.codi.base_model.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.base_model.model.lm_head.weight.clone())

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
