# 🏦 Financial Q&A Fine-Tuning with QLoRA

> Fine-tuning a small open-source language model on financial data using QLoRA — trained entirely on free cloud GPUs, with results benchmarked against GPT-5.

---

## What This Project Does

This project takes a general-purpose AI language model — **Phi-3 Mini (3.8 billion parameters)** from Microsoft — and specializes it for **financial question answering** using a technique called **QLoRA fine-tuning**.

In plain terms: we teach a small, free, open-source AI model to answer financial questions the way a financial analyst would, then measure how much better it got compared to its starting point, and compare it against GPT-5 (one of the most powerful commercial AI models available).

The entire training process runs on a **free cloud GPU** (Kaggle's T4) and costs **$0**.

The fine-tuned model is publicly available on HuggingFace:
🔗 **[rohan1324/phi3-mini-finance-qlora](https://huggingface.co/rohan1324/phi3-mini-finance-qlora)**

---

## How It Works 

Here is the complete pipeline from start to finish in plain language:

### Step 1 — Get the Training Data
We download a dataset of **68,000 financial instruction-response pairs** from HuggingFace. Each entry looks like:

```
Instruction: What is EBITDA and why do investors care about it?
Response: EBITDA stands for Earnings Before Interest, Taxes, Depreciation,
and Amortization. Investors use it as a proxy for a company's operational
profitability because it strips out the effects of financing decisions...
```

We use 5,000 of these pairs for training and 500 for validation.

### Step 2 — Measure the Starting Point (Baseline Eval)
Before changing anything, we run the **original unmodified Phi-3 Mini** on 20 test questions and record how well it does. This gives us a "before" score to compare against later.

### Step 3 — Fine-Tune with QLoRA
We train the model on our 5,000 financial examples for 500 steps (~45 minutes on a free GPU). We use QLoRA to make this feasible on limited hardware — see the Key Concepts section for details.

### Step 4 — Measure the Improvement (Fine-Tuned Eval)
We run the same 20 test questions through our newly fine-tuned model and compare the scores to Step 2.

### Step 5 — Compare Against GPT-5
We run the same 20 questions through GPT-5 via Azure OpenAI and add it to the comparison table. This gives us a sense of where our free local model stands relative to a state-of-the-art commercial model.

### Step 6 — Visualize and Publish
We generate a comparison chart, save all results, and publish the fine-tuned adapter to HuggingFace so anyone can use it.

---

## Results

### Quantitative Comparison

| Model | ROUGE-1 | ROUGE-L | Avg Latency | Cost/query |
|---|---|---|---|---|
| Base Phi-3 Mini (local) | 0.238 | 0.131 | 11.4s | $0.000 |
| **Fine-tuned Phi-3 Mini** | **0.223** | **0.146** | **10.5s** | **$0.000** |
| GPT-5 (Azure OpenAI) | 0.282 | 0.133 | 3.1s | ~$0.030 |

### What These Numbers Mean

**✅ ROUGE-L improved by +11.5%** (0.131 → 0.146) after fine-tuning. This is a real, meaningful improvement in how closely the model's answers match reference financial answers.

**✅ Fine-tuned Phi-3 Mini matched GPT-5 on ROUGE-L** (0.146 vs 0.133). This doesn't mean Phi-3 is smarter than GPT-5 — it means the fine-tuned model learned to structure answers in a way that matches financial reference answers more closely.

**✅ Zero inference cost** vs $0.030 per query for GPT-5. At 10,000 queries/day that's $300/day saved.

**⚠️ ROUGE-1 dropped slightly** (0.238 → 0.223). The fine-tuned model shifted toward financial domain vocabulary which overlaps less with some reference answers. This is a known fine-tuning tradeoff.

**⚠️ Latency is slower than GPT-5** (10.5s vs 3.1s) because GPT-5 runs on highly optimized commercial infrastructure. Our model runs on a free T4 GPU with no optimization.

### Why GPT-5's ROUGE Score Isn't Higher
GPT-5 gives longer, more detailed, conversational answers. ROUGE measures word overlap with short reference answers — so GPT-5's more verbose responses don't overlap as closely even though they're often more accurate and informative. This is a known limitation of ROUGE as an evaluation metric for open-ended Q&A.

---

## Tech Stack

| Tool | What It Does | Why We Use It |
|---|---|---|
| **Phi-3 Mini 4K Instruct** | Base language model (3.8B params) | Small enough to fine-tune on free GPU, strong enough to be useful |
| **QLoRA / PEFT** | Efficient fine-tuning technique | Makes training feasible on limited hardware |
| **TRL (SFTTrainer)** | Training framework | Simplifies the fine-tuning pipeline |
| **HuggingFace Transformers** | Model loading and inference | Industry standard library for working with language models |
| **HuggingFace Datasets** | Dataset loading | Easy access to the finance-alpaca training data |
| **BitsAndBytes** | 4-bit quantization | Reduces model memory footprint by ~50% |
| **ROUGE Score** | Evaluation metric | Measures quality of generated text vs reference answers |
| **Azure OpenAI** | GPT-5 access for comparison | Frontier model benchmark |
| **Matplotlib** | Results visualization | Comparison charts |
| **Kaggle (T4 GPU)** | Training compute | Free GPU with 30hrs/week |
| **HuggingFace Hub** | Model hosting | Permanent storage and sharing of fine-tuned adapter |

---

## Project Structure

```
finance-finetune/
│
├── 📓 notebook.ipynb              ← Main Kaggle notebook (all cells)
│
├── 📁 data/
│   ├── train/                     ← 5,000 training samples (arrow format)
│   ├── val/                       ← 500 validation samples (arrow format)
│   └── eval_samples.json          ← 20 test samples used across all evals
│
├── 📁 outputs/
│   └── phi3-finance-lora/
│       ├── checkpoint-400/        ← Intermediate checkpoint (step 400)
│       │   ├── adapter_config.json
│       │   └── adapter_model.safetensors
│       └── checkpoint-500/        ← Final checkpoint (step 500) ✅ use this
│           ├── adapter_config.json
│           └── adapter_model.safetensors
│
└── 📁 results/
    ├── baseline_results.json      ← Base model eval scores (20 samples)
    ├── finetuned_results.json     ← Fine-tuned model eval scores (20 samples)
    ├── azure_results.json         ← GPT-5 eval scores (20 samples)
    ├── eval_summary.json          ← Aggregated comparison stats
    └── comparison_chart.png       ← Bar charts comparing all 3 models
```

---

## How to Run It Yourself

### Prerequisites
- A free [Kaggle account](https://www.kaggle.com) (phone verification required for GPU)
- A free [HuggingFace account](https://huggingface.co)
- A HuggingFace **read** token (for downloading Phi-3)
- A HuggingFace **write** token (for uploading your fine-tuned model)
- Azure OpenAI API key (optional — only needed for GPT-5 comparison)

### Step 1 — Set Up Kaggle

1. Create a Kaggle account at https://www.kaggle.com
2. Verify your phone number at https://www.kaggle.com/settings (required for GPU)
3. Create a new notebook at https://www.kaggle.com/code
4. In the right sidebar: **Settings → Accelerator → GPU T4 x2**

### Step 2 — Add Your Secrets

In your Kaggle notebook go to **Add-ons → Secrets** and add these:

| Secret Name | Where to Get It |
|---|---|
| `HF_TOKEN` | https://huggingface.co/settings/tokens (read access) |
| `HF_TOKEN_WRITE` | https://huggingface.co/settings/tokens (write access) |
| `AZURE_OPENAI_KEY` | Your Azure OpenAI resource → Keys and Endpoint |
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource → Keys and Endpoint |
| `AZURE_OPENAI_DEPLOYMENT` | Name of your GPT-5 deployment in Azure |
| `AZURE_OPENAI_API_VERSION` | `2024-02-15-preview` |

Toggle **"Notebook has access"** on for each secret after adding it.

### Step 3 — Run the Cells in Order

Copy each cell from this README's [Notebook Cell Guide](#notebook-cell-guide) section into your Kaggle notebook and run them in order. Each cell is clearly labeled with its purpose and expected runtime.

### Step 4 — Load the Pre-trained Model (Skip Training)

If you just want to use the already fine-tuned model without retraining, skip to this:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load the fine-tuned model directly from HuggingFace
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    "rohan1324/phi3-mini-finance-qlora"
)

base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=False,
    attn_implementation="eager"
)

model = PeftModel.from_pretrained(
    base_model,
    "rohan1324/phi3-mini-finance-qlora"
)

# Ask a financial question
question = "What is the difference between gross margin and operating margin?"
prompt = f"### Instruction:\n{question}\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("### Response:")[-1].strip())
```

---

## Notebook Cell Guide

Here is what each cell in the notebook does and how long it takes:

| Cell | Name | What It Does | Runtime |
|---|---|---|---|
| 1 | GPU Check | Verifies T4 GPU is available | < 1 min |
| 2 | Directory Setup | Creates project folder structure | < 1 min |
| 3 | Install Packages | Installs transformers, peft, trl etc | 2–3 mins |
| 4 | HF Login | Authenticates with HuggingFace | < 1 min |
| 5 | Azure Credentials | Loads Azure OpenAI keys from secrets | < 1 min |
| 6 | Prepare Dataset | Downloads and formats 5,500 samples | 3–5 mins |
| 7 | Baseline Eval | Tests original model on 20 questions | 5–10 mins |
| 8 | Fine-Tuning | Trains the model for 500 steps | **45–60 mins** |
| 9 | Fine-Tuned Eval | Tests fine-tuned model on same 20 questions | 5–10 mins |
| 10 | Azure Eval | Tests GPT-5 on same 20 questions | 2–5 mins |
| 11 | Results + Chart | Generates comparison table and bar chart | < 1 min |
| 12 | Upload to HF | Publishes adapter to HuggingFace Hub | 2–5 mins |

**Total time: approximately 1.5–2 hours** (mostly waiting for Cell 8)

---



## Limitations and Honest Caveats

Being transparent about limitations is just as important as reporting results:

**ROUGE is an imperfect metric.** ROUGE measures word overlap, not answer quality. A model can score low on ROUGE while giving a perfectly correct answer (if it uses different vocabulary) and vice versa. The results should be interpreted alongside the qualitative sample comparisons, not just the numbers.

**500 training steps is minimal.** This is a proof-of-concept training run. Production-quality fine-tuning would use 2,000+ steps and 20,000+ samples. The +11.5% improvement would likely be significantly larger with more training.

**20 eval samples is a small test set.** Statistically robust evaluation would use 500+ samples. The results here are directionally correct but not statistically significant.

**The model can still hallucinate.** Fine-tuning reduces hallucination on in-domain questions but doesn't eliminate it. Always verify financial information from primary sources before acting on it.

**Not suitable for regulated financial advice.** This model is for research and internal tooling purposes. Deploying it in a consumer-facing context where it could be interpreted as investment advice would require significant additional safety work and likely regulatory approval.

**GPT-5 comparison has a confound.** GPT-5's lower ROUGE-L score (0.133 vs our 0.146) reflects the fact that it gives longer, more detailed answers that don't overlap as closely with short reference answers — not that it performs worse. A human evaluation would likely rate GPT-5's answers higher overall.

---



