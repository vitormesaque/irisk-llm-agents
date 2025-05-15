# iRisk+: Scalable Risk Analysis from App Reviews Using LLM Agents

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

App reviews contain rich insights about user experience and software quality, but their volume and unstructured nature make manual analysis impractical. Traditional risk matrices help prioritize issues based on severity and likelihood, but they do not scale to meet the dynamic and high-frequency nature of app feedback.

**iRisk+** is an end-to-end, open-source framework that automates the extraction, classification, and prioritization of app-related risks using large language models (LLMs). The system is composed of two key components:

- **i-LLaMA-Agent**: A real-time agent that collects and classifies new app reviews on demand, based on user-defined instructions.
- **i-LLaMA**: A fine-tuned model designed for retrospective analysis, capable of detecting recurring issues and identifying temporal risk trends across app versions.

Unlike traditional approaches that depend on manual tagging and predefined rules, iRisk+ supports **fully automated**, **scalable**, and **privacy-compliant** risk analysis.

### Key Features

- Automatic ingestion of user reviews  
- Risk classification based on severity and probability  
- Dynamic risk matrix construction  
- Interactive dashboards for monitoring and decision-making

iRisk+ empowers QA, product, and development teams to detect regressions early, prioritize critical issues, and align maintenance decisions with real user feedback.

---
## Online Demo
- [iRisk](https://irisk-live.mappidea.com)

## Index Terms
- Opinion Mining
- Large Language Model
- App Reviews
- Risk Matrix
- Issue Prioritization


## Introduction

iRisk is a scalable microservice designed to classify issue risks based on crowdsourced app reviews. By leveraging the power of Large Language Models (LLMs), specifically the fine-tuned LLaMA 3 model, iRisk can effectively analyze app reviews, identify potential issues, and prioritize them using a risk matrix. This tool aims to facilitate better decision-making, monitoring, and risk mitigation for app developers and stakeholders.

## Features

- **Automated Review Analysis**: Uses the fine-tuned LLaMA 3 model to analyze app reviews and extract relevant issues.
- **Risk Matrix Generation**: Automatically generates a risk matrix to help prioritize maintenance actions.
- **Scalable Architecture**: Implemented as a microservice, allowing multiple containers to run concurrently for efficient processing.
- **Dashboard and Visualizations**: Provides an automated dashboard with visualizations for easy decision-making.

## Getting Started

### Prerequisites

- Docker
- OLLAMA

### Ollama
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

1. Configure the repository

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
 ```
2. Install the NVIDIA Container Toolkit packages
```bash
   sudo apt-get install -y nvidia-container-toolkit
```
3. Configure Docker to use Nvidia driver
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

4. Start the container
```bash
 docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/vitormesaque/irisk.git
    cd irisk
    ```

2. Create an OLLAMA container with the fine-tuned LLaMA 3 model:
    ```bash
    ollama create irisk -f ./Modelfile
    ```

### Usage

To run the iRisk microservice, execute the following command:
```bash
ollama run irisk
```


# i-LLAMA: Fine-Tuning iRisk using Unsloth with LLAMA3 and LoRA Adapter

This section describes the process of fine-tuning the iRisk tool using Unsloth with LLAMA3 and LoRA Adapter, leveraging a dataset of app reviews with identified and prioritized issues.

## What is LoRA Adapter?

LoRA (Low-Rank Adaptation) is a technique used to efficiently fine-tune large language models. Instead of updating all parameters of a model during fine-tuning, LoRA inserts low-rank matrices into each layer of the transformer architecture. This approach significantly reduces the computational cost and memory requirements, making fine-tuning more accessible and faster while maintaining high performance.

## What is Unsloth?

Unsloth is a fine-tuning framework designed for scaling the adaptation of large language models. It integrates with modern machine learning pipelines to enable seamless and efficient training. Unsloth supports various adapters, including LoRA, to optimize and enhance the performance of models like LLAMA3 for specific tasks, such as issue detection and prioritization from user reviews.

## Fine-Tuning with Unsloth and LoRA Adapter

The following steps outline how to fine-tune iRisk using the dataset of app reviews and identified issues, utilizing Unsloth with LLAMA3 and LoRA Adapter:

### 1. Prepare the Dataset

Ensure your dataset is in JSON format, with examples from domains such as 'Music & Audio', 'Entertainment', 'Communication', 'Shopping', 'Social Media', and 'Travel & Local' apps. The dataset should include reviews with and without issues, and the output should be in JSON format for each identified issue.

### 2. Set Up the Environment

Install the necessary dependencies:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
pip install datasets
```

### 3. Load and Fine-Tune the Model

#### Load the Model

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

#### Load the Dataset

```python
from datasets import load_dataset
dataset = load_dataset("vitormesaque/irisk", split = "train")
```

#### Apply LoRA Adapter

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
```

### 4. Data Preparation

Format the dataset with appropriate prompts:

```python
irisk_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = irisk_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("vitormesaque/irisk", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)
```

### 5. Train the Model

Set up the trainer and start training:

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()
```

### 6. Inference

Generate responses using the fine-tuned model:

```python
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    irisk_prompt.format(
        "Extract issues from the user review in JSON format. For each issue, provide: label functionality, severity (1-5), likelihood (1-5), category (Bug, User Experience, Performance, Security, Compatibility, Functionality, UI, Connectivity, Localization, Accessibility, Data Handling, Privacy, Notifications, Account Management, Payment, Content Quality, Support, Updates, Syncing, Customization), and the sentence.",
        "I used to love this app, but now it's become frustrating as hell. We can't see lyrics, we can't CHOOSE WHAT SONG WE WANT TO LISTEN TO, we can't skip a song more than a few times, there are ads after every two songs, and all in all it's a horrible overrated app. If I could give this 0 stars, I would.",
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 512)
```


# i-LLAMA Client

This is a Python package to extract issues from user reviews using an API.

## Installation

You can install the package using pip:

```bash
pip install git+https://github.com/vitormesaque/illama_client.git
```


## Usage


### 1. Define the review text

Create a string variable to hold the review text.

```python
review = """It's slow to load and crashes often. The GPS is also inaccurate, showing the driver at the wrong location."""
```

### 2. Import necessary modules

Import the required functions and libraries.

```python
from illama_client.issue_extractor import extract_issues
import json
import pandas as pd
```

### 3. Extract issues from the review

Use the `extract_issues` function to extract issues from the review. Replace `'api'` and `'model'` with your appropriate values.

```python
# Replace 'api' and 'model' with your actual values
issues = extract_issues(review, api, model)
```

### 4. Convert the extracted issues to a DataFrame

Parse the JSON response and normalize it into a pandas DataFrame.

```python
# Convert the JSON response to a pandas DataFrame
json_data = json.loads(issues)
df = pd.json_normalize(json_data, 'issues')
```

### 5. Display the DataFrame

Print the DataFrame to see the extracted issues in a tabular format.

```python
# Display the DataFrame
print(df)
```

### Example Output

The above code will extract the mentioned issues in the review and format them into a pandas DataFrame. Here's an example of the JSON output and the resulting DataFrame:

#### JSON Output

```json
{
  "review": "It's slow to load and crashes often. The GPS is also inaccurate, showing the driver at the wrong location.",
  "issues": [
    {
      "label": "Slow Loading",
      "functionality": "App Speed",
      "severity": 3,
      "likelihood": 5,
      "category": "Performance",
      "sentence": "It's slow to load."
    },
    {
      "label": "Crashes",
      "functionality": "App Stability",
      "severity": 4,
      "likelihood": 5,
      "category": "Bug",
      "sentence": "The app crashes often."
    },
    {
      "label": "GPS Inaccuracy",
      "functionality": "Navigation Accuracy",
      "severity": 3,
      "likelihood": 4,
      "category": "Functionality",
      "sentence": "The GPS is also inaccurate."
    }
  ]
}
```

#### DataFrame Output

```plaintext
           label       functionality  severity  likelihood     category                sentence
0   Slow Loading           App Speed         3           5  Performance         It's slow to load.
1        Crashes        App Stability         4           5          Bug       The app crashes often.
2  GPS Inaccuracy  Navigation Accuracy         3           4  Functionality  The GPS is also inaccurate.
```



## Resources
- [i-LLAMA Client](https://github.com/vitormesaque/illama-client)
- [LoRA Adapter on Hugging Face](https://huggingface.co/vitormesaque/lora_model)
- [Unsloth Documentation](https://github.com/Unsloth/unsloth)

For detailed steps and configurations, please refer to the official documentation of Unsloth and the Hugging Face transformers library.

## Authors

- **Vitor Mesaque Alves de Lima**  
  Três Lagoas Campus (CPTL) Federal University of Mato Grosso do Sul (UFMS), Três Lagoas, Brazil  
  [vitor.lima@ufms.br](mailto:vitor.lima@ufms.br)

- **Jacson Rodrigues Barbosa**  
  Institute of Informatics (INF) Federal University of Goiás (UFG), Goiânia, Brazil  
  [jacson@inf.ufg.br](mailto:jacson@inf.ufg.br)

- **Ricardo Marcodes Marcacini**  
  Institute of Mathematics and Computer Sciences (ICMC) University of São Paulo (USP), São Carlos, Brazil  
  [ricardo.marcacini@usp.br](mailto:ricardo.marcacini@usp.br)


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE for details.



