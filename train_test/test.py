from datasets import load_dataset
from random import randrange
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, ConversationalPipeline, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, logging, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from huggingface_hub import notebook_login, login
from dotenv import load_dotenv

model_id = "/home/hice1/wxu372/scratch/workspace/llama-2-7B-4bit-python-coder/merged_model"

# Get the type
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)
merged_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache = False, device_map={"": 0}, cache_dir = './.cache')
merged_model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


import pandas as pd

results = []


prompt = f""""### Instruction: Can you tell me some information about: 

### Input: ECE 6601 

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)

generated_instruction = f"{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"

results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
print(f"\nGround truth:\nECE 6601 at Georgia Institute of Technology, commonly referred to as Georgia Tech, is a course titled \"Random Processes\". This course is designed to develop the theoretical framework for the processing of random signals and data. It offers 3.000 credit hours and consists of 3.000 lecture hours. The course is not applicable for the CMPE (Computer Engineering) or EE (Electrical Engineering) degrees, and does not include any lab hours. Recent professors who have taught this course include Geoffrey Li, Min Luo, and Guotong Zhou, indicating a diverse range of expertise contributing to the course content")

prompt = f"""### Instruction: Can you tell me some information about:

### Input: ECE4783: Introduction to Medical Image Processing

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)

generated_instruction = f"{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"

results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: the curriculum of the System and Controls track

### Response:
"""


input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)

generated_instruction = f"{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"

results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: which semester(s) I can enroll in ECE6280

### Response:
"""


input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)

generated_instruction = f"{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"

results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: the course ECE 6456: Solar Cells

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)

generated_instruction = f"{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"

results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: making an appointment with Professor Larry Heck for course-related questions

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: the contact information of Professor Larry Heck for course-related questions

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: the technical interest groups for a freshman in ECE

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: when ECE6001 is taught

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: asking Professor Larry Heck a quick question without an appointment

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: ECE4783: Introduction to Medical Image Processing

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: the curriculum of the System and Controls track

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: which semester(s) I can enroll in ECE6280

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: the course ECE 6456: Solar Cells

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: making an appointment with Professor Larry Heck for course-related questions

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: the contact information of Professor Larry Heck for course-related questions

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: the technical interest groups for a freshman in ECE

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: when ECE6001 is taught

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: asking Professor Larry Heck a quick question without an appointment

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


prompt = f"""### Instruction: Can you tell me some information about:

### Input: the difficulty level of ECE 7252: Advanced Signal Processing Theory

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")

prompt = f"""### Instruction: Can you tell me some information about:

### Input: eligibility of a sophomore for ECE Seminar (ECE 2002)

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")

prompt = f"""### Instruction: Can you tell me some information about:

### Input: professors to contact for a thesis in Bioengineering

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
results.append({
        "Generated Instruction": generated_instruction
    })

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


df = pd.DataFrame(results)
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

filename = f"generated_course_instructions_v3_{current_time}.csv"

df.to_csv(filename, index=False)



