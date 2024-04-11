# Tech-Enhanced-AI-Interview-Learning-Platform

## Techshila Chatbot Interviewer

The Chatbot Interviewer is a conversational AI system designed to engage in natural language interactions with users, simulating the experience of an interview. The chatbot is powered by the Mistral language model, which has been fine-tuned to provide relevant and engaging questions based on the user's responses. The model also gives basic corrections in the grammers of the Candidate using Fine Tuned Mistral.

![Pipeline_Flowchart](https://github.com/beingamanforever/Tech-Enhanced-AI-Interview-Learning-Platform/blob/main/Techshila%202024.png)

## Table of Contents

- Introduction
- Features
- Dataset and Model
- Installation
- Usage
- Customization
- Contributing
- License
  

## Features
Seamless conversation flow with the user
Relevant and context-aware questions
Customizable question templates and interview flow
User-friendly interface for easy interaction
Integration with various platforms (e.g., web, mobile, messaging apps)
Installation
To set up the Chatbot Interviewer, follow these steps:

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
!python app.py
  </code>
</pre>

## Usage
To use the Chatbot Interviewer, simply interact with the chatbot through the provided interface. The chatbot will ask relevant questions based on the user's responses, simulating a natural interview experience.

## Customization
You can customize the Chatbot Interviewer by modifying the following:

- Question templates: Adjust the pre-defined question templates to suit your specific needs.
- Interview flow: Modify the logic that determines the sequence and selection of questions.
- User interface: Customize the look and feel of the chatbot interface to match your branding and design.

## Contributing
We welcome contributions to the Chatbot Interviewer project. If you'd like to contribute, please follow these steps:

- Fork the repository
- Create a new branch for your feature or bug fix
- Make your changes and commit them
- Push your changes to your forked repository
- Submit a pull request
