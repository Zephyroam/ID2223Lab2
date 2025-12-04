# ID2223Lab2
This repository contains the implementation and evaluation results for KTH ID2223 Lab 2 of our group: Xiangyu Shi and Ali Banaei.

### Deliverables

* Source code: this GitHub repository

* Results of Task 2: `README.md`

* Public UI: Hugging Face Spaces URL: https://huggingface.co/spaces/Zephyroam/KTHID2223Lab2

### Experimental Setup
#### Base Models

We fine-tuned and evaluated the following open-source LLMs:

`unsloth/Llama-3.2-1B-Instruct`, `unsloth/Llama-3.2-3B-Instruct`, `unsloth/Qwen2.5-3B-Instruct`.

#### Fine-tuning with LoRA

We employed LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of the base models. [FineTome Instruction Dataset](https://huggingface.co/datasets/mlabonne/FineTome-100k) is used as the training data to enhance instruction-following and dialogue style generalization.

As a model-centric approach to further improve model performance, we experimented with different LoRA ranks, 8, 16, and 32, to observe their impact on the fine-tuned models.


#### Benchmarks

We evaluated the fine-tuned models on the following benchmarks:
- GSM8K: math reasoning accuracy.
- IFEval: instruction-following score.
- MMLU: multi-domain general knowledge accuracy.

Higher is better across all metrics.

### Results

| Model                 | LoRA Rank | Data     | GSM8K | IFEval | MMLU   |
|-----------------------|-----------|----------|-------|--------|--------|
| Llama-3.2-1B-Instruct | 8         | FineTome | 0.22  | **0.27**   | 0.4172 |
| Llama-3.2-1B-Instruct | 16        | FineTome | **0.24**  | 0.19   | 0.4239 |
| Llama-3.2-1B-Instruct | 32        | FineTome | 0.22  | 0.21   | **0.4295** |
| Llama-3.2-3B-Instruct | 8         | FineTome | 0.53  | 0.37   | 0.5561 |
| Llama-3.2-3B-Instruct | 16        | FineTome | 0.50  | 0.43   | 0.5579 |
| Llama-3.2-3B-Instruct | 32        | FineTome | **0.56**  | **0.44**   | **0.5612** |
| Qwen2.5-3B-Instruct   | 8         | FineTome | **0.60**  | **0.26**   | 0.6453 |
| Qwen2.5-3B-Instruct   | 16        | FineTome | 0.57  | 0.21   | **0.6470** |
| Qwen2.5-3B-Instruct   | 32        | FineTome | 0.59  | 0.23   | 0.6468 |

We observe that different LoRA ranks yield varying performance across benchmarks. For instance, a LoRA rank of 16 achieves the highest MMLU score for both Llama-3.2-1B and Qwen2.5-3B models, while a rank of 32 performs best on GSM8K for Llama-3.2-3B. This indicates that the optimal LoRA rank may depend on the specific task and model architecture.

### Hugging Face Model Storage
The fine-tuned models are publicly available on Hugging Face Model Hub:
- [Llama-3.2-1B-Instruct, Rank 8](https://huggingface.co/Zephyroam/llama-3.2-1b-instruct-unsloth-bnb-16bit-FineTome-r8)
- [Llama-3.2-1B-Instruct, Rank 16](https://huggingface.co/Zephyroam/llama-3.2-1b-instruct-unsloth-bnb-16bit-FineTome-r16)
- [Llama-3.2-1B-Instruct, Rank 32](https://huggingface.co/Zephyroam/llama-3.2-1b-instruct-unsloth-bnb-16bit-FineTome-r32)
- [Llama-3.2-3B-Instruct, Rank 8](https://huggingface.co/Zephyroam/llama-3.2-3b-instruct-unsloth-bnb-16bit-FineTome-r8)
- [Llama-3.2-3B-Instruct, Rank 16](https://huggingface.co/Zephyroam/llama-3.2-3b-instruct-unsloth-bnb-16bit-FineTome-r16)
- [Llama-3.2-3B-Instruct, Rank 32](https://huggingface.co/Zephyroam/llama-3.2-3b-instruct-unsloth-bnb-16bit-FineTome-r32)
- [Qwen2.5-3B-Instruct, Rank 8](https://huggingface.co/Zephyroam/qwen-2.5-3b-instruct-unsloth-bnb-16bit-FineTome-r8)
- [Qwen2.5-3B-Instruct, Rank 16](https://huggingface.co/Zephyroam/qwen-2.5-3b-instruct-unsloth-bnb-16bit-FineTome-r16)
- [Qwen2.5-3B-Instruct, Rank 32](https://huggingface.co/Zephyroam/qwen-2.5-3b-instruct-unsloth-bnb-16bit-FineTome-r32)

### Chatbot UI
We have developed a public chatbot UI hosted on Hugging Face Spaces, allowing users to interact with our fine-tuned models. The UI supports the following features:
- Model Selection: Users can choose between the three fine-tuned models: Llama-3.2-1B, Llama-3.2-3B, and Qwen2.5-3B.
- Conversation History: The UI maintains a history of the conversation, enabling users to keep track of the dialogue context.
- Hyperparameter Configuration: Users can adjust parameters such as temperature, top-p, and max tokens to customize the response generation.

### Code Structure
- `fine_tune.py`: Script for fine-tuning models using LoRA.
- `eval.py`: Script for evaluating fine-tuned models on benchmarks.
- `app.py`: Gradio application for the chatbot UI.


