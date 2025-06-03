import os
import shutil

from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from mcts_ttt_rewards import *

SYSTEM_PROMPT = """
You are a powerful gaming agent who can make proper decisions to beat the user in gaming tasks.
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{}
</reasoning>
<answer>
{}
</answer>
"""


def main():
    # Load up `Llama 3.1 8B Instruct`, and set parameters
    max_seq_length = 2048  # Can increase for longer reasoning traces
    lora_rank = 32  # Larger rank = smarter, but slower

    os.environ["WANDB_PROJECT"] = "mcts-ttt"

    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name="unsloth/Meta-Llama-3.1-8B",
    #     max_seq_length=max_seq_length,
    #     dtype=None,  # None for auto-detection.
    #     load_in_4bit=True,  # False for LoRA 16bit
    #     fast_inference=True,  # Enable vLLM fast inference
    #     max_lora_rank=lora_rank,
    #     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    # )

    # Load in SFT model
    shutil.unpack_archive("mcts_ttt_sft_lora.zip", "sft_lora_model")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="sft_lora_model",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        fast_inference=True,  # Enable vLLM fast inference
    )
    # FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=lora_rank,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )

    # Prepare GRPO training data
    def grpo_formatting_prompts_func(examples):
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            for prompt in examples["prompt"]
        ]
        # print(messages)
        texts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in
                 messages]
        return {"prompt": texts, }

    def get_mcts_ttt_grpo(stage="grpo", split="train") -> Dataset:
        path = "mcts_ttt_test.jsonl"
        if split == "train":
            path = "mcts_ttt_train_" + stage + ".jsonl"
        data = load_dataset("json", data_files=path, split="train")
        data = data.map(grpo_formatting_prompts_func, batched=True, )
        return data  # type: ignore

    grpo_dataset = get_mcts_ttt_grpo()

    max_prompt_length = 350

    training_args = GRPOConfig(
        learning_rate=5e-6,
        # lr_scheduler_type = "cosine",
        optim="paged_adamw_8bit",
        logging_steps=10,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,  # Increase to 4 for smoother training
        num_generations=6,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        num_train_epochs=2,
        # max_steps = 250,
        save_steps=250,
        # max_grad_norm = 0.1,
        report_to="wandb",  # Can use Weights & Biases
        output_dir="grpo_outputs",
        # scale_rewards = False, # Disables std scaling, recommended by Dr. GRPO paper
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            # strict_format_reward_func,
            soft_format_reward_func,
            xmlcount_reward_func,
            # strict_reasoning_format_reward_func,
            soft_reasoning_format_reward_func,
            move_format_reward_func,
            mcts_reward_func,
            optimality_reward_func,
        ],
        args=training_args,
        train_dataset=grpo_dataset,
    )
    trainer.train()

    # Save model
    model.save_pretrained("grpo_lora_model")
    tokenizer.save_pretrained("grpo_lora_model")

    shutil.make_archive("mcts_ttt_grpo_lora", "zip", "grpo_lora_model")


if __name__ == "__main__":
    main()
