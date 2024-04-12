# Tech-Enhanced-AI-Interview-Learning-Platform

## Chatbot Interviewer 

The Chatbot Interviewer is a conversational AI system designed to engage in natural language interactions with users, simulating the experience of an interview. The chatbot is powered by the Mistral language model, which has been fine-tuned to provide relevant and engaging questions based on the user's responses. The model also gives basic corrections in the grammers of the Candidate using Fine Tuned Mistral.

## Project Pipeline Diagram  :open_file_folder:

![Pipeline_Flowchart](https://github.com/beingamanforever/Tech-Enhanced-AI-Interview-Learning-Platform/blob/main/Techshila%202024.png)

>[!TIP]
> The pipeline diagram of the whole project, can be glanced at [here](https://www.figma.com/file/W9jITgwibMiuyctGrFvMmA/Techshila-2024?type=whiteboard&node-id=0%3A1&t=BuFpFkRdg49CU8vH-1) , for the pdf version [click here](https://drive.google.com/file/d/1qedcxfMOqoqeLeiodIBy9gVwta--fIry/view?usp=drive_link)

## Table of Contents :bar_chart:

- Introduction
- Features
- Audio To Text
- Model Training & Fine Tuning
- Deployment
- Hyperparameters
- Installation
- Video Demo
- Usage
- Customization
- Contributing
- License
  

## Features :tophat:

- Seamless conversation flow with the user
- Relevant and context-aware questions
- Customizable question templates and interview flow
- User-friendly interface for easy interaction
- Integration with various platforms (e.g., web, mobile, messaging apps)
- Diversified catering of domains

# Audio To Text Conversion :headphones:
We used [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) transcription with [CTranslate2](https://github.com/OpenNMT/CTranslate2/) , as the backbone for our Automatic Speech to text conversion. This implementation is up to 4 times faster than [openai/whisper](https://github.com/openai/whisper) for the same accuracy while using less memory.
### Calculate Speaking Pace

We then wrote a function to calculate the speaking pace, which would return a value which would then passed on to pace checker, which would then give feedback to the user.
```python
def calculate_speaking_pace(transcription, chunk_length):
    """
    Calculate speaking pace in words per second.

    Args:
    transcription (str): The transcription of the spoken text.
    chunk_length (int): The length of the audio chunk in seconds.

    Returns:
    float: Speaking pace in words per second.
    """
    words = transcription.split()
    num_words = len(words)
    speaking_rate = num_words / chunk_length  # Words per second
    return speaking_rate
``` 
### Pace Checker
We set a optimal pace range (hyperparameter), within which answers would be acceptable, else the user would be given an prompt to ”Too Fast” or ”Too Slow”.
```python
def pace_checker(pace):
    optimal_pace_range = (1, 3)
    if optimal_pace_range[0] <= pace <= optimal_pace_range[1]:
        print("Good pace")
    elif pace < optimal_pace_range[0]:
        print("Very slow")
    elif pace > optimal_pace_range[1]:
        print("Too fast")
```
### Speech to Text
Then we converted the corrected audio data captured from the user into text.
```python
def get_text():
    audio_path = "audio.wav"
    result = model_audio.transcribe(audio_path)
    segments, info = result
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    text = ""
    for segment in segments:
        text += segment.text
    pace = calculate_speaking_pace(text, chunk_length=DEFAULT_CHUNK_LENGTH)
    print(pace)
    pace_checker(pace)
    return text
```
# Model Training & Fine Tuning :game_die:
## Finetuning of Mistral 7B LLM using QLoRA
### Preparing Data for Fine tuning

- Data Collection: Collected various datasets (namely SDE-Data, DS-Data, PMConsult data, Combined data)
- Data Cleaning: Removed noise, dropped columns, corrected errors, and ensuring a uniform format, then combined all these datasets and used it for fine tuning purposes of the LLM.
- Pushed dataset to Huggingface Hub
- Loading the Datasets:
```python
import pandas as pd
from transformers import DataCollatorForLanguageModeling

df1 = pd.read_excel('/kaggle/input/dataset -qa/combined_dataset.xlsx')
df2 = pd.read_excel('/kaggle/input/dataset -qa/Sde_data.xlsx')
df3 = pd.read_excel('/kaggle/input/dataset -qa/DS_data.xlsx')
df4 = pd.read_excel('/kaggle/input/dataset -qa/PMConsult_data.xlsx')

df1 = df1.rename(columns={'Position/Role': 'Job_Position'})
df = pd.concat([df1, df2, df3, df4], ignore_index=True)
df.drop(columns=['Column1.6', 'Column1.7'], inplace=True)

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

df.to_csv('data.csv', index=False)
```
- Data Splitting: Splitted the dataset into two parts train and test with a split percentage of 80% for training, 10% for validation, and 10% for testing.
### Fine-Tuning the Model — The Crux
We did fine-tuning the [Mistral 7B model](https://www.kdnuggets.com/mistral-7b-v02-fine-tuning-mistral-new-open-source-llm-with-hugging-face) using the [QLoRA](https://github.com/artidoro/qlora) (Quantization and LoRA) method. This approach combines quantization and [LoRA adapters](https://docs.vllm.ai/en/latest/models/lora.html) to improve the model’s performance. We used the [PEFT](https://huggingface.co/blog/peft) library from [Hugging Face](https://huggingface.co/) to facilitate the fine-tuning process. 
- Loading the dependencies:
>[!NOTE]
> You only have to run this once per machine
````python
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U datasets scipy
````
```python
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```
- Loading the Pre-trained Model: The [Mistral 7B model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) is loaded into the deep learning framework & we set up the accelerator using the [FullyShardedDataParallelPlugin](https://huggingface.co/docs/accelerate/v0.11.0/en/fsdp) & Accelerator.
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM

# We would be fine tuning Mistral-7B LLM
base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)
```
- Tokenization: Tokenized the text data into the format suitable for the model to ensures com- patibility with the pre-trained architecture. We use self-supervised fine-tuning to align the label & input ids.
```python
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True
)
tokenizer.pad_token = tokenizer.eos_token
```
```python
def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenizer.pad_token = tokenizer.eos_token
```
- Set up LoRA: Now, we prepare the model for fine-tuning by applying LoRA adapters to the linear layers of the model.
- Custom Data Collator & Question Generator: Created custom data loaders for training, validation, & testing. These loaders facilitate efficient model training by feeding data in batches, enabling the model to learn from the dataset effectively. Provided custom prompts based on the job position, to generate questions
```python
def generate_and_tokenize_prompt(df):
    full_prompt = f"""
    You are now conducting an interview for the {df['Job Position']} role.
    You have asked the candidate the following question: {df['Question']}
    The candidate has responded as follows: {df['Answer']}
    Please formulate a thoughtful follow-up question to probe deeper into the
    candidate’s understanding and experience of the candidate’s
    response in relation to the desired skills and knowledge for the {df['Job Position']} role.
    """
    return tokenize(full_prompt)
```
>[!TIP]
> After this we splitted the combined dataset and the uploaded it on hugging-face, and then mapped our prompt to the dataset for the fine tuning of the model, so as to increase the efficiency.
```python
from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("akshatshaw/eddcv")
```
```python
tokenized_train_dataset = dataset['train'].map(generate_and_tokenize_prompt)
tokenized_val_dataset = dataset['test'].map(generate_and_tokenize_prompt)
```
- Trainable parameters & Fine Tuning Loop: Set the training loop for the model to fine tune, after which the model would be outputting the evaluation prompt.
  
![image](https://github.com/beingamanforever/Tech-Enhanced-AI-Interview-Learning-Platform/assets/121532863/902bed7d-992e-4bc6-ab6e-360fdf2be56f)
>[!NOTE]
> Model Training parameter graphs can be accessed [here](https://wandb.ai/akshatshaw-iitr/huggingface/reports/FineTuning-mistral-model--Vmlldzo3NDk2ODQ3?accessToken=sp6lqo5qjh405qm3zuo1jhr2wnmdqqvaj03ejxyz4txoj8pweileedlh3kxk33vm)
```python
# Now, we prepare the model for fine-tuning by applying LoRA adapters to the model
def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * (trainable_params / all_param)}"
    )
```
>[!TIP]
> Reference about usage of [peft](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/nemo_megatron/peft/landing_page.html) from [LoRA’s ](https://arxiv.org/pdf/2106.09685.pdf)paper here here After training, you can use the fine-tuned model for inference. You’ll need to load the base Mistral model from the Huggingface Hub and then load the [QLoRA adapters](https://github.com/artidoro/qlora/blob/main/README.md) from the best-performing checkpoint directory.

### Model Evaluation
[BLeU score](https://en.wikipedia.org/wiki/BLEU) (Bi-Lingual Evaluation Understudy) is a metric for automatically evaluating machine- translated text. The BLEU score is a number between zero and one that measures the similarity of the machine-translated text to a set of high quality reference translations.

![image](https://github.com/beingamanforever/Tech-Enhanced-AI-Interview-Learning-Platform/assets/121532863/22706b3f-f4bf-42dc-ba6f-3068f4ff49fb)

## Evaluation Metrics Scores

<pre>
  <code>
  BLEU Score: {'bleu': 0.023421683258425006, 'precisions': [0.10455943307928868, 0.030399418366883264, 0.013347545998053482, 0.007093204709887928], 'brevity_penalty': 1.0, 'length_ratio': 4.421083743842365, 'translation_length': 22437, 'reference_length': 5075}
    
  Rouge Score:  {'rouge1': 0.1828892019214882, 'rouge2': 0.05595716756740922, 'rougeL': 0.14946545570101497, 'rougeLsum': 0.14924759519559638}
  </code>
</pre>

### Mistral 7B Instruct V02 model:
<pre>
  <code>
BLEU Score 0.02 
Rouge 1-gram Score 0.18
  </code>
</pre>
### Phi2 model: [Refer this repo for FT-Phi2 notebook](https://github.com/AGAMPANDEYY/Techshila_Agam_Notebooks/blob/main/phi-2-peft.ipynb)
<pre>
  <code>
Rouge 1-gram Score 0.38
Rouge L-gram Score 0.34
  </code>
</pre>

Although, the Fine tuned Phi2 model had a good Rouge score compared to Mistral-7B, Phi2 being a SLM couldn't perform well in understanding context and gave unsatisfactory summary and selection-decision of the candidate's interview.

Therefore, the team chose to use Mistral 7B Instruct model. 

*Challenges with Scores:*

- Lack of dataset quantity limits the reproducibility of the fine tuned model to same length and context as the evaluation dataset
- Lesser epoch for training due to lesser GPU access on Kaggle and other GPU enabled notebooks

# Deployment :clapper:
We uploaded our training datasets and Fine Tuned Mistral-LLM (mistralai/Mistral-7B-Instruct-v0.2) using Quantized-LoRA to hugging-face.
## Frontend
Tech-Stack used - HTML/ CSS
- We created a website interface page where the user would select the interview domain according to which he would be questioned.
- This page would lead to the chatbot page, where the user can interact with our model by means of audio and textual data

![image](https://github.com/beingamanforever/Tech-Enhanced-AI-Interview-Learning-Platform/assets/121532863/71bd53f9-086d-46b3-9c8f-1f79e43f57a0)
## Backend
Tech-Stack used - Javascript / Flask
### Flask
- In the flask file, two models are imported - first one is faster-wisper, second one is Mistral.
- After receiving the audio files from the user, it is saved in the device locally in .wav format.
- Then we read this file using faster-wisper and then convert it into text.
- We then feed this text into our fine tuned mistral model, which then asks the user questions based on the domain selected by the user.
- Our model also corrects the grammatical errors present in the text prompted by the user.
- Model then outputs a desired answer based on the previous inputs given by the user and the context
  
# Hyperparameters
These are the list of Hyperparameters involved and how they were tuned:
- n-bits quantisation: we used 4-bit quantisation to speed up the process, and make our model light as it reduces the precision of the weights and biases involved in the model. For more info [click here](https://arxiv.org/pdf/2106.09685.pdf).
- Optimal Speech Range: we set the range to be (1-3) words/sec, as the average speaking rate of a human is 2.5 words/sec, if the input is not within this range then the user is prompted to adjust.
- peft LoRA configuration: ”lora-alpha” this sets the value of the alpha parameter used in LoRA. It controls the strength of the linearity assumption in LoRA & ”r” this parameter controls the number of LoRA blocks in the model.
- Number of Epochs for training of model: We set the number of epochs to be 50, but the eval/loss was plateauing near about 35 epochs. The training paramters graph can be [seen here](https://wandb.ai/akshatshaw-iitr/huggingface/reports/FineTuning-mistral-model--Vmlldzo3NDk2ODQ3?accessToken=sp6lqo5qjh405qm3zuo1jhr2wnmdqqvaj03ejxyz4txoj8pweileedlh3kxk33vm)


## Dataset Collection and Model Fine-Tuning

The datasets are stored on Huggingface Hub [HF_Dataset](AgamP/techshila_ml) and Github Repo dataset [Refer this](https://github.com/OmdenaAI/omdena-hyderabad-Chatbot-for-interview.git)
The synthetic data generated were done using prompts given to ChatGPT as .csv files, which was then pushed to HF. This is the format of the training dataset.
<pre>
  <code>
DatasetDict
    ({
    train: Dataset
    ({
        features: ['Job_Position', 'Question', 'Answer', 'Interview Phase', 'Answer Quality', 'Job Position'],
        num_rows: 1742
    })
    test: Dataset({
        features: ['Job_Position', 'Question', 'Answer', 'Interview Phase', 'Answer Quality', 'Job Position'],
        num_rows: 581
    })
})
  </code>
</pre> 

Base Model Taken for training was [Mistral-7b-Instruct-V02](akshatshaw/mistral-interview-finetune) which was *fine tuned on custom dataset using QLora with 50 epochs.* 
[Refer this for WanB report](https://github.com/beingamanforever/Tech-Enhanced-AI-Interview-Learning-Platform/blob/main/FineTuning%20mistral%20model%20_%20huggingface%20%E2%80%93%20Weights%20%26%20Biases.pdf))

# Installation
To set up the Chatbot Interviewer, follow these steps:

## Clone the repository:
<pre>
  <code class="python">
git clone https://github.com/your-username/chatbot-interviewer.git
  </code>
</pre>

## Install the required dependencies:
<pre>
  <code class="python">
cd chatbot-interviewer
pip install -r requirements.txt
  </code>
</pre>

## Fine-tune the Mistral model with your custom data:
<pre>
  <code>
python fine_tune_model.py
  </code>
</pre>

## Start the chatbot server:

<pre>
  <code>
python app.py
  </code>
</pre>

# Video Demo
Video Demo can be accessed from [here](https://drive.google.com/file/d/10UcOz291u-XWc1RXnXWgn0KzAkuLFwHM/view)

## Customization :video_game:
You can customize the Chatbot Interviewer by modifying the following:

- Question templates: Adjust the pre-defined question templates to suit your specific needs.
- Interview flow: Modify the logic that determines the sequence and selection of questions.
- User interface: Customize the look and feel of the chatbot interface to match your branding and design.
- More Diversified Domains: The present number of domains can be extended.


## Contributing :open_hands:
We welcome contributions to the Chatbot Interviewer project. 
>[!NOTE]
> If you'd like to contribute, please follow these steps:

- Fork the repository
- Create a new branch for your feature or bug fix
- Make your changes and commit them
- Push your changes to your forked repository
- Submit a pull request
  
## Contributers :rocket:
[Agam Pandey](https://github.com/AGAMPANDEYY) , [Akshat Shaw](https://github.com/akshatshaw) , [Aman Behera](https://github.com/beingamanforever) , [Kushagra Singh](https://github.com/git-kush) , [Vaibhav Prajapati](https://github.com/vaibhavprajapati-22) , [Vishnu jain](https://github.com/Vishnuujain)
