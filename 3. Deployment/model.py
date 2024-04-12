from peft import PeftModel, PeftConfig
import peft
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
from transformers import BitsAndBytesConfig

from huggingface_hub import login

login(token="your_hf_api_key")

base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True)

tokenizer.pad_token = tokenizer.eos_token

config = PeftConfig.from_pretrained("akshatshaw/mistral-interview-finetune")
model = PeftModel.from_pretrained(base_model, "akshatshaw/mistral-interview-finetune")

# eval_prompt = """
# You are now conducting an interview for the Customer Service Representative role.
# You have asked the candidate the following question: Name any Two Improvements You Made in the Previous Company?
# The candidate has responded as follows: As a few of my team members were late to work, which impacted the work, I changed the login conditions by introducing timely login incentives. Hence, it improved the login time.‚Äù There were a few people who used to miss the calls and sometimes used break aux and stayed on it for longer. Announcing a reward for the highest number of calls attended improved the work balance effectively
# Please formulate a thoughtful follow-up question to probe deeper into the candidate's understanding and experience of the candidate's response in relation to the desired skills and knowledge for the Customer Service Representative role.
# """

# model_input = eval_tokenizer(eval_prompt, return_tensors="pt")
#
# model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=200)[0], skip_special_tokens=True))

#
def generate_output(prompt):
    max_new_tokens = 200
    model_input = tokenizer(prompt, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        output = model.generate(**model_input, max_new_tokens=max_new_tokens)

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    return decoded_output