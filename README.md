# **Data Leakage in Vision-Language Models (VLMs)**  

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Project Status](https://img.shields.io/badge/status-Active-brightgreen)]()  
[![GitHub Issues](https://img.shields.io/github/issues/atharvahude/VLM-Data-Leakage-Study)]()  

## 📌 **Overview**  
This repository contains our research on **data leakage in Vision-Language Models (VLMs)**. We investigate whether VLMs are genuinely solving problems or merely retrieving stored knowledge, given that training datasets are often undisclosed by developers. Our study explores test-train set leakage and proposes fair evaluation frameworks.  

## 📑 **Problem Statement**  
Recent advancements in language models have shown exceptional performance, sometimes outperforming larger counterparts. However, concerns arise regarding whether models achieve this through actual problem-solving or by retrieving memorized data. While prior studies have examined leakage in Large Language Models (LLMs), **investigations into VLMs remain limited**. Our research focuses on detecting and evaluating data leakage in VLMs.  

## 🔬 **Approach**  
We devised multiple strategies to analyze potential data leakage:  

1. **Question + Answer**: Checking if the model reconstructs a full QA pair.  
2. **Question Only**: Evaluating if the model can predict missing parts of a question.  
3. **Relevant Information About a Question**: Identifying if the model has seen helpful data.  
4. **Very Similar Questions**: Testing responses to paraphrased/restructured versions.  
5. **Very Similar Tasks**: Exploring model performance on similar QA tasks.  
6. **Seen Bias?**: Detecting biased outputs in different conditions.  
7. **Baseline**: Setting a control accuracy by testing with original images and questions.  

## 🏗 **Methodology**  
We conducted experiments across multiple models using different datasets and techniques:  

- **Models Tested**:  
  - Qwen-7B-VL  
  - Llava-1.5-7B  
  - Microsoft Phi-3-Vision-128k-Instruct  
  - LLaMA 3.2 11B Vision Instruct & Pretrained  

- **Datasets Used**:  
  - ScienceQA  
  - VQAv2  
  - MathVista  
  - ChartQA  

- **Techniques Applied**:  
  - **Paraphrasing & Restructuring** (GPT-4o, LLaMA-3 70B)  
  - **Token Removal** (using NLTK)  
  - **Image Modifications** (flipping, rotation, masking)  
  - **Rouge Metrics** (for overlap detection)  

## 📊 **Results Summary**  
The following are key findings from our experiments:  

| **Model**  | **Dataset**  | **Baseline Accuracy (%)** | **Key Observations** |
|------------|-------------|----------------|----------------|
| **Qwen-7B-VL**  | ScienceQA | 76.29  | Accuracy drops with image flipping & truncation |
| **Llava-1.5-7B** | VQAv2 | 64 | No major leakage detected |
| **Phi-3-Vision-128k** | MathVista | 34.45 | Image removal drops accuracy to **8.13%** |
| **LLaMA 3.2 11B Instruct** | ChartQA | 73.34 | Masking tokens reveals potential leakage |
| **LLaMA 3.2 11B Pretrained** | ScienceQA | 30 | Model avoids image-based responses |

Full experimental results are available in our report.  

## 📈 **Key Findings**  
- Data leakage is not always obvious but can manifest in **paraphrased** and **masked input** cases.  
- Models exhibit **biases**, especially when visual data is absent.  
- **Llama 3.2 models** occasionally reconstruct masked tokens accurately, hinting at memorization.  
- **Phi-3-Vision struggles without visual input**, showing a high dependency on images.  
- **Qwen-7B-VL misidentifies flipped/rotated images**, indicating weaknesses in robustness.  

## 👥 **Team Members**  
- **Atharva Hude** - Microsoft Phi-3 Vision on MathVista  
- **Sonal Prabhu** - LLaMA 3.2 11B on AI2D & ChartQA  
- **Shrinidhi Kumbhar** - ScienceQA testing for LLaMA 3.2  
- **Shazeb Ata** - Llava model testing on VQAv2  
- **Shivam Singh** - Qwen-7B-VL testing on ScienceQA  

## 🔗 **Resources & References**  
📄 **Report**: [Final Report](https://github.com/atharvahude/VLM-Data-Leakage-Study)  
📖 **Citations**:  
1. [2304.08485](https://arxiv.org/abs/2304.08485)  
2. [2404.18824](https://arxiv.org/pdf/2404.18824)  
3. [NAACL-424](https://aclanthology.org/2024.naacl-long.424.pdf)  

## 🚀 **Future Work**  
- Expanding dataset sizes for **stronger statistical validation**  
- Investigating **prompt engineering** to reveal hidden biases  
- Exploring **adversarial attacks** for further robustness testing  

---


