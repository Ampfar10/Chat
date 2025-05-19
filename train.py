import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

DATA_FILE = "conversations.json"
MODEL_DIR = "trained_model"

def make_dataset_file():
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    with open("chat.txt", "w") as f:
        for pair in data:
            f.write(f"User: {pair['input']}\nBot: {pair['response']}\n")

def train_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="chat.txt",
        block_size=64
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_total_limit=1,
        logging_steps=10,
        save_steps=50,
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

make_dataset_file()
train_model()
