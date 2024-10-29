import sys
import logging

import datasets
from peft import LoraConfig, get_peft_model
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from model import ClipPhi3Model
from dataset import ImageConversationDataset
from trainer import MultimodalTrainer

logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
CLIP_EMBED = 512  # Adjust based on your CLIP model
PHI_EMBED = 3072  # Adjust based on the phi-3-mini model


training_config = {
    "bf16": True,
    "do_eval": False,
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 100,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": 12000,
    "output_dir": "./checkpoint_dir",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 16,
    "per_device_train_batch_size": 16,
    "remove_unused_columns": False,
    "save_steps": 200,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
    }

peft_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["o_proj", "qkv_proj"]
}

bnb_config = {
    "load_in_4bit":True,
    "bnb_4bit_quant_type":"nf4",
    "bnb_4bit_compute_dtype":"bfloat16",
    "bnb_4bit_use_double_quant":True,
}

train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)
bnb_conf = BitsAndBytesConfig(**bnb_config)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
    + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")
logger.info(f"PEFT parameters {peft_conf}")

model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization_config=bnb_conf,
    device_map={"": 0},
    use_cache=False,
    attn_implementation='eager', 
)

phi_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
phi_model = get_peft_model(phi_model, peft_conf)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'
tokenizer.chat_template = "{% for message in messages %}{% if message['from'] == 'system' %}{{'<|system|>' + message['value'] + '<|end|>'}}{% elif message['from'] ==\
 'human' %}{{'<|user|>' + message['value'] + '<|end|>'}}{% elif message['from'] == 'gpt' %}{{'<|assistant|>' + message['value'] +\
 '<|end|>'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}"

model = ClipPhi3Model(phi_model, CLIP_EMBED, PHI_EMBED, 'projections_model.pth')

dataset = load_dataset('json', data_files='llava_instruct_150k.json', split='train')

def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["conversations"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example

processed_train_dataset = dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=['image', 'conversations'],
    desc="Applying chat template to train_sft",
)

dataset = ImageConversationDataset(processed_train_dataset)
train_set, val_set = torch.utils.data.random_split(dataset, [0.9,0.1])
val_set = torch.utils.data.dataset.Subset(val_set, range(int(0.20*len(val_set))))

def collate_fn(batch):
    image_embeds = torch.stack([item['image_embeds'] for item in batch])
    conversations_texts = [item['text'] for item in batch]

    batch_size = image_embeds.shape[0]
    num_image_tokens = image_embeds.shape[1]
    
    image_tokens = torch.full((batch_size, num_image_tokens), -100, dtype=torch.long)

    conversations_tokens = tokenizer(conversations_texts, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    
    conversations_ids = conversations_tokens['input_ids']
    conversations_mask = conversations_tokens['attention_mask']

    input_ids = torch.cat([image_tokens, conversations_ids], dim=1)
    conversations_mask = torch.cat([torch.ones((batch_size, num_image_tokens), dtype=torch.long), conversations_mask], dim=1)

    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # Set the last token's label to -100 (ignored in loss calculation)

    return {
        "conversations_ids": conversations_ids,
        "image_embeds": image_embeds,
        "conversations_mask": conversations_mask,
        "labels": labels
    }

trainer = MultimodalTrainer(model=model,
                            args=train_conf,
                            train_dataset=train_set,
                            eval_dataset=val_set,
                            data_collator=collate_fn,
                            tokenizer=tokenizer)

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

tokenizer.padding_side = 'left'
metrics = trainer.evaluate()
metrics["eval_samples"] = len(val_set)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

model.save_pretrained(train_conf.output_dir)