
HuggingFace：https://huggingface.co/MedAIBase/AntAngelMed

ModelScope：https://modelscope.cn/models/MedAIBase/AntAngelMed

Github: https://github.com/MedAIBase/AntAngelMed/tree/main


# Introduction

**AntAngelMed is Officially Open Source! 🚀**

**AntAngelMed**, developed by **Ant Group** and the **Health Commission of Zhejiang Province**, is the largest and most powerful open-source medical language model to date.

# Core Highlights

+ 🏆**World-leading performance on authoritative benchmarks**: AntAngelMed surpasses all open-source models and a range of top proprietary models on OpenAI's HealthBench, and ranks first overall on the Chinese authority benchmark MedAIBench.
+ 🧠**Advanced Medical Capabilities**: AntAngelMed achieves its professional medical capabilities through a rigorous three-stage training pipeline: continual pre-training on medical corpora, supervised fine-tuning with high-quality instructions, and GRPO-based reinforcement learning. This process equips the model with deep medical knowledge, sophisticated diagnostic reasoning, and robust adherence to safety and ethics.
+ ⚡**Extremely efficient inference:** Leveraging [Ling-flash-2.0](https://arxiv.org/abs/2507.17702)’s high-efficiency MoE, AntAngelMed matches the performance of ~40B dense models while activating only 6.1B parameters of its 100B parameters. It achieves over 200 tokens/s on H20 hardware and supports 128K context length.

# **📊** Benchmark Results

## **HealthBench**

[**HealthBench**](https://arxiv.org/abs/2505.08775) is an open-source medical evaluation benchmark released by OpenAI, designed to assess the performance of Large Language Models (LLMs) in real-world medical environments through highly simulated multi-turn dialogues. AntAngelMed achieved outstanding performance on this benchmark, ranking first among all open-source models, with a particularly significant advantage on the challenging HealthBench-Hard subset.

## **MedAIBench**

[**MedAIBench**](https://www.medaibench.cn) is an authoritative medical LLM evaluation system developed by the National Artificial Intelligence Medical Industry Pilot Facility. AntAngelMed also **ranks first overall** and demonstrates strong comprehensive professionalism and safety, especially in medical knowledge Q&A and medical ethics/safety.

![](https://github.com/MedAIBase/AntAngelMed/blob/main/Figures/Figure%20%7C%20HealthBench-medAIBench.png)
**Figure | AntAngelMed ranks first among open-source models on HealthBench and first on MedAIBench**

## **MedBench**

[**MedBench**](https://arxiv.org/abs/2511.14439) is a scientific and rigorous benchmark designed to evaluate LLMs in the Chinese healthcare domain. It comprises 36 independently curated evaluation datasets and covers approximately 700,000 samples. AntAngelMed ranks first on the MedBench self-assessment leaderboard and leads across five core dimensions: medical knowledge question answering, medical language understanding, medical language generation, complex medical reasoning, and safety and ethics, highlighting the model's professionalism, safety, and clinical applicability.

![](https://github.com/MedAIBase/AntAngelMed/blob/main/Figures/Figure%20%7C%20AntAngelMed%20ranks%20first%20on%20the%20MedBench%20self-assessment%20leaderboard.png)
**Figure | AntAngelMed ranks first on the MedBench self-assessment leaderboard.**



# 🔧 Technical Features

## Professional three-stage training pipeline

AntAngelMed employs a carefully designed three-stage training process to deeply integrate general capabilities with medical expertise:

+ **Continual Pre-Training:** Based on Ling-flash-2.0, AntAngelMed is continually pre-trained with large-scale, high-quality medical corpora (encyclopedias, web text, academic publications), injecting profound domain and world knowledge.
+ **Supervised Fine-Tuning (SFT):** A multi-source and heterogeneous high-quality instruction dataset is constructed at this stage. General data (math, programming, logic) strengthen core chain-of-thought capabilities of AngAngel, while medical scenarios (doctor–patient Q&A, diagnostic reasoning, safety/ethics) provide deep adaptation for improved clinical performance.
+ **Reinforcement Learning (RL):** Using the [**GRPO**](https://arxiv.org/pdf/2402.03300) algorithm and task-specific reward models, RL precisely shapes model behavior—emphasizing empathy, structural clarity, and safety boundaries, and encouraging evidence-based reasoning on complex cases to reduce hallucinations and improve accuracy.

![](https://github.com/MedAIBase/AntAngelMed/blob/main/Figures/Figure%20%7C%20Professional%20three-stage%20training%20pipeline.jpg)

**Figure | Professional three-stage training pipeline**

## Efficient MoE architecture with high-speed inference

AntAngelMed inherits Ling-flash-2.0’s advanced design. Guided by [Ling Scaling Laws](https://arxiv.org/abs/2507.17702), the model uses a **1/32 activation-ratio MoE** and is comprehensively optimized across core components, including expert granularity, shared expert ratio, attention balance, no auxiliary loss + sigmoid routing, MTP layer, QK-Norm, and Partial-RoPE.

These refinements enable **small-activation** MoE models to deliver up to **7× efficiency** over similarly sized dense architectures. In other words, with only 6.1B activated parameters, AntAngelMed can match ~40B dense model performance. Because of its small activated parameter count, AntAngelMed offers substantial speed advantages:

+ On H20 hardware, inference exceeds **200 tokens/s**—about **3× faster** than a 36B dense model.
+ With **YaRN extrapolation**, it supports a **128K context length**; as output length grows, relative speedups can reach 7× or more.

![Figure | Model Architecture Diagram (https://huggingface.co/inclusionAI/Ling-flash-2.0)](https://github.com/MedAIBase/AntAngelMed/blob/main/Figures/Figure%20%7C%20Model%20Architecture%20Diagram.png)

We have also specifically optimized AntAngelMed for inference acceleration by employing **FP8 quantization combined with EAGLE3 optimization**. Under a concurrency of 32, this approach significantly boosts inference throughput compared to using FP8 alone, with improvements of **71% on HumanEval, 45% on GSM8K**, and **as high as 94% on Math-500**. This achieves a robust balance between inference performance and model stability.


# Quickstart

## 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with transformers:

```plain
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "MedAIBase/AntAngelMed"  # model_id or your_local_model_path
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "What should I do if I have a headache?"
messages = [
    {"role": "system", "content": "You are AntAngelMed, a helpfull medical assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt", return_token_type_ids=False).to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## 🤖 ModelScope

If you're in mainland China, we strongly recommend you to use our model from 🤖 [ModelScope](https://modelscope.cn/organization/MedAIBase).

## Deployment - on Nvidia A100

### vLLM

vLLM supports offline batched inference or launching an OpenAI-Compatible API Service for online inference.

#### Environment Preparation

Please prepare the following environment:

```plain
pip install vllm==0.11.0
```

#### Inference

```plain
from modelscope import AutoTokenizer
from vllm import LLM, SamplingParams

def main():
    model_path = "MedAIBase/AntAngelMed" # model_id or your_local_model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.05,
        max_tokens=16384,
    )
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=4, 
    )

    prompt = "What should I do if I have a headache?" 
    messages = [
        {"role": "system", "content": "You are AntAngelMed, a helpfull medical assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = llm.generate([text], sampling_params)
    print(outputs[0].outputs[0].text)
  
if __name__ == "__main__":
    main()
```

### **SGLang**

#### **Environment Preparation**

Prepare the following environment:

```plain
pip install sglang -U
```

You can use Docker image as well:

```plain
docker pull lmsysorg/sglang:latest
```

#### **Run Inference**

BF16 and FP8 models are supported by SGLang, it depends on the dtype of the model in ${MODEL_PATH}. They both share the same command in the following:

+ Start server:

```plain
SGLANG_USE_MODELSCOPE=true python -m sglang.launch_server \
    --model-path $MODLE_PATH \
    --host 0.0.0.0 --port $PORT \
    --trust-remote-code \
    --attention-backend fa3 \
    --tensor-parallel-size 4 \
    --served-model-name AntAngelMed
```

+ Client:

```plain
curl -s http://localhost:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What should I do if I have a headache?"}]}'
```

More usage can be found [here](https://docs.sglang.ai/basic_usage/send_request.html).

## **Deployment - on Ascend 910B**

### **vLLM-Ascend**

vLLM-Ascend (vllm-ascend) is a community-maintained hardware backend that enables vLLM to run on Ascend NPUs.

#### **Environment Preparation**

We recommend using the 64*8GB memory version of the Ascend Atlas 800I A2 server to run this model.

We recommend using Docker for deployment. Please prepare the environment by following the steps below:

```plain
docker pull quay.io/ascend/vllm-ascend:v0.11.0rc3
```

Next, you can start and join the container by running the following commands, then proceed with subsequent operations inside the container.

```plain
NAME=your container name
MODEL_PATH=put your absolute model path here if you already have it locally.

docker run -itd --privileged --name=$NAME --net=host \
 --shm-size=1000g \
   --device /dev/davinci0 \
   --device /dev/davinci1 \
   --device /dev/davinci2\
   --device /dev/davinci3 \
   --device /dev/davinci4 \
   --device /dev/davinci5 \
   --device /dev/davinci6 \
   --device /dev/davinci7 \
   --device=/dev/davinci_manager \
   --device=/dev/hisi_hdc \
   --device /dev/devmm_svm \
   -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
   -v /usr/local/sbin:/usr/local/sbin \
   -v /etc/hccn.conf:/etc/hccn.conf \
   -v $MODEL_PATH:$MODEL_PATH \
   quay.io/ascend/vllm-ascend:v0.11.0rc2 \
   bash

docker exec -u root -it $NAME bash
```

For both offline and online inference with vLLM, ensure the following environment variables are configured in the terminal before execution:

```plain
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"
export NPU_MEMORY_FRACTION=0.97
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=100
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#You can use ModelScope mirror to speed up download:
export VLLM_USE_MODELSCOPE=true
```

#### **Offline Inference**

```plain
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
model_path = "MedAIBase/AntAngelMed" # model_id or your_local_model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=16384)
llm = LLM(model=model_path, 
    dtype='float16',
    tensor_parallel_size=4,                     
    gpu_memory_utilization=0.97,  
    enable_prefix_caching=True,  
    enable_expert_parallel=True,
    trust_remote_code=True)
prompt = "What should I do if I have a headache?"
messages = [
    {"role": "system", "content": "You are AntAngelMed, a helpful medical assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
outputs = llm.generate([text], sampling_params)
```

#### **Online Inference**

```plain
model_id=MedAIBase/AntAngelMed
taskset -c 0-23 python3 -m vllm.entrypoints.openai.api_server \
  --model $model_id \
  --max-num-seqs=200 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --enable_expert_parallel \
  --gpu_memory_utilization 0.97 \
  --served-model-name AntAngelMed \
  --max-model-len 32768 \
  --port 8080 \
  --enable-prefix-caching \
  --block-size 128 \
  --async-scheduling \
  --trust_remote_code
```

```plain
curl http://0.0.0.0:8080/v1/chat/completions -d '{
    "model": "AntAngelMed",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What should I do if I have a headache?"
        }
    ],
    "temperature": 0.6
}'

```

For detailed guidance, please refer to the vLLM-Ascend [here](https://docs.vllm.ai/projects/ascend/zh-cn/latest/quick_start.html).

# License

This code repository is licensed under [the MIT License](https://github.com/inclusionAI/Ling-V2/blob/master/LICENCE).
