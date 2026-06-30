from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base")


dataset = load_dataset("HuggingFaceTB/smoltalk2_everyday_convs_think")

config = SFTConfig(
    output_dir="./smollm3-finetuned",
    per_device_train_batch_size=4,
    learning_rate=5e5,
    max_steps=1000,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    args=config
)

trainer.train()