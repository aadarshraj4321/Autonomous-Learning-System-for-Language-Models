import os
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,)

from peft import LoraConfig, PeftModel, get_peft_model
from trl import DPOTrainer



BASE_MODEL_NAME = "EleutherAI/gpt-neo-125M"
DATASET_REPO = "boyinfuture/autonomous-learning-dataset"
NEW_ADAPTER_REPO = "boyinfuture/autonomous-learning-125m-adapter"


def train_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Base model '{BASE_MODEL_NAME}' loading")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False


    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_size = "right"


    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
    )


    print(f"Dataset '{DATASET_REPO}' loading")
    dataset = load_dataset(DATASET_REPO)


    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1,
        max_steps=-1,
        report_to="tensorboard",
        save_steps=50,
    )



    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=0.1,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=512,
        max_length=1024,
    )


    print("DPO training starting")
    dpo_trainer.train()
    print("Training completed")


    print(f"New Adapter '{NEW_ADAPTER_REPO}' pushes")
    dpo_trainer.model.push_to_hub(NEW_ADAPTER_REPO)
    tokenizer.push_to_hub(NEW_ADAPTER_REPO)
    print("Adapter successfully pushed")




if __name__ == "__main__":
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("Hugging Face token not found. `huggingface-cli login` do login")
    else:
        train_model()
