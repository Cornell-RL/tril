import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

inputs = torch.load('test.pt')
gen_kwargs = {
    'do_sample': False,
    'top_k': 0.0,
    'top_p': 1.0,
    'max_new_tokens': 53,
    'min_new_tokens': 53,
    'eos_token_id': None,
    'pad_token_id': None,
}
peft_config = LoraConfig(r=1024, lora_alpha=2048, lora_dropout=0.0)

tokenizer = AutoTokenizer.from_pretrained('models/sft_model_2.9')
model = AutoModelForCausalLM.from_pretrained('models/sft_model_2.9')
model = model.cuda(0)
inputs = {k: v.cuda(0) for k, v in inputs.items()}


model.generation_config.pad_token_id = None
model.generation_config.eos_token_id = None
print("Pre Peft: ", model.generation_config)
pre_peft_out = model.generate(**inputs, **gen_kwargs, generation_config=model.generation_config, use_cache=True)

model = get_peft_model(model, peft_config=peft_config)
print("Post Peft: ", model.generation_config)
model = model.cuda(1)
inputs = {k: v.cuda(1) for k, v in inputs.items()}
post_peft_out = model.generate(**inputs, **gen_kwargs, generation_config=model.generation_config, use_cache=True)


print(f"====== Pre Peft EOS: {torch.any(pre_peft_out == tokenizer.eos_token_id)}")
print(f"====== Post Peft EOS: {torch.any(post_peft_out == tokenizer.eos_token_id)}")

import pdb; pdb.set_trace()
