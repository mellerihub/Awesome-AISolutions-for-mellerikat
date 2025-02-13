# Awesome AI Solutions for mellerikat 

---
<font size=5><center><b> Table of Contents </b> </center></font>
- [Awesome Solutions](#awesome-solutions)
  - [Multimodal Object Detection](#multimodal-object-detection)
  - [For Tabular Data](#solutions-for-tabular-data)
  - [For Timeseries Data](#solutions-for-timeseries-data)
  - [Foundation Model Based](#multimodal-in-context-learning)
  - [Others](#others)
- [Datasets](#awesome-datasets)
  - [Datasets of Pre-Training for Alignment](#datasets-of-pre-training-for-alignment)
  - [Datasets of Multimodal Instruction Tuning](#datasets-of-multimodal-instruction-tuning)
  - [Datasets of In-Context Learning](#datasets-of-in-context-learning)
  - [Datasets of Multimodal Chain-of-Thought](#datasets-of-multimodal-chain-of-thought)
  - [Datasets of Multimodal RLHF](#datasets-of-multimodal-rlhf)
  - [Benchmarks for Evaluation](#benchmarks-for-evaluation)
  - [Others](#others-1)
---

# Awesome Solutions

## Multimodal Object Detection
|  Title  |   Solution builder  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Framework](https://img.shields.io/badge/ALO-V2.7.0-blue) ![Star](https://img.shields.io/github/stars/IDEA-Research/GroundingDINO.svg?style=social&label=Star) <br> [**Zero-Shot Object Detection**](https://github.com/mellerihub/Awesome-AISolutions-for-mellerikat/blob/main/docs/zeroshot_objectdetection.md) <br> | LGE | 2025-01-26 | [Github](https://github.com/QwenLM/Qwen2.5-VL) | [Demo](https://huggingface.co/spaces/Qwen/Qwen2.5-VL) |
| ![Framework](https://img.shields.io/badge/ALO-V2.7.0-blue) ![Star](https://img.shields.io/github/stars/baichuan-inc/Baichuan-Omni-1.5.svg?style=social&label=Star) <br> [**One-Shot Object Detection**](https://github.com/baichuan-inc/Baichuan-Omni-1.5/blob/main/baichuan_omni_1_5.pdf) <br> | LGE | 2025-01-26 | [Github](https://github.com/baichuan-inc/Baichuan-Omni-1.5) | Local Demo |
| ![Framework](https://img.shields.io/badge/ALO-V2.7.0-blue)  <br> [**Labeling Automation for Object Detection**](https://arxiv.org/pdf/2412.10360) <br> | LGE | 2024-12-13 | - | - |


## Solutions For Tabular Data
|  Title  |   Soution builder  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Framework](https://img.shields.io/badge/ALO-V2.7.0-blue) ![Star](https://img.shields.io/github/stars/mellerikat-aicontents/Tabular-Classification-Regression.svg?style=social&label=Star) <br> [**Tabular-Classification-Regression**](https://github.com/mellerihub/Awesome-AISolutions-for-mellerikat/blob/main/docs/tcr.md) <br> | LGE | 2024-02-16 | [Github](https://github.com/mellerikat-aicontents/Tabular-Classification-Regression) | - |
| ![Framework](https://img.shields.io/badge/ALO-V2.7.0-blue) ![Star](https://img.shields.io/github/stars/mellerikat-aicontents/Tabular-Anomaly-Detection.svg?style=social&label=Star) <br> [**Tabular-Anomaly-Detection**](https://github.com/mellerihub/Awesome-AISolutions-for-mellerikat/blob/main/docs/tad.md) <br> | LGE | 2024-02-02 | [Github](https://github.com/mellerikat-aicontents/Tabular-Anomaly-Detection) | - |
| ![Framework](https://img.shields.io/badge/ALO-V2.7.0-blue) ![Star](https://img.shields.io/github/stars/mellerikat-aicontents/Tabular-Anomaly-Detection.svg?style=social&label=Star) <br> [**Tabular-Anomaly-Detection**](https://github.com/mellerihub/Awesome-AISolutions-for-mellerikat/blob/main/docs/tad.md) <br> | POSTECH | 2024-02-02 | [Github](https://github.com/mellerikat-aicontents/Tabular-Anomaly-Detection) | - |
| ![Framework](https://img.shields.io/badge/ALO-V2.7.0-blue) ![Star](https://img.shields.io/github/stars/mellerikat-aicontents/Tabular-Anomaly-Detection.svg?style=social&label=Star) <br> [**Tabular-Anomaly-Detection**](https://github.com/mellerihub/Awesome-AISolutions-for-mellerikat/blob/main/docs/tad.md) <br> | POSTECH | 2024-02-02 | [Github](https://github.com/mellerikat-aicontents/Tabular-Anomaly-Detection) | - |
| ![Framework](https://img.shields.io/badge/ALO-V2.7.0-blue) ![Star](https://img.shields.io/github/stars/mellerikat-aicontents/Tabular-Anomaly-Detection.svg?style=social&label=Star) <br> [**Tabular-Anomaly-Detection**](https://github.com/mellerihub/Awesome-AISolutions-for-mellerikat/blob/main/docs/tad.md) <br> | POSTECH | 2024-02-02 | [Github](https://github.com/mellerikat-aicontents/Tabular-Anomaly-Detection) | - |


## Solutions For Timeseries Data
|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/dongyh20/Insight-V.svg?style=social&label=Star) <br> [**Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models**](https://arxiv.org/pdf/2411.14432) <br> | arXiv | 2024-11-21 | [Github](https://github.com/dongyh20/Insight-V) | - |
| ![Star](https://img.shields.io/github/stars/ggg0919/cantor.svg?style=social&label=Star) <br> [**Cantor: Inspiring Multimodal Chain-of-Thought of MLLM**](https://arxiv.org/pdf/2404.16033.pdf) <br> | arXiv | 2024-04-24 | [Github](https://github.com/ggg0919/cantor) | Local Demo |
| ![Star](https://img.shields.io/github/stars/deepcs233/Visual-CoT.svg?style=social&label=Star) <br> [**Visual CoT: Unleashing Chain-of-Thought Reasoning in Multi-Modal Language Models**](https://arxiv.org/pdf/2403.16999.pdf) <br> | arXiv | 2024-03-25 | [Github](https://github.com/deepcs233/Visual-CoT) | Local Demo |


## Foundation Models
|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/DAMO-NLP-SG/VideoLLaMA3.svg?style=social&label=Star) <br> [**VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding**](https://arxiv.org/pdf/2501.13106) <br> | arXiv | 2025-01-22 | [Github](https://github.com/DAMO-NLP-SG/VideoLLaMA3) | [Demo](https://huggingface.co/spaces/lixin4ever/VideoLLaMA3) |
| ![Star](https://img.shields.io/github/stars/baaivision/Emu3.svg?style=social&label=Star) <br> [**Emu3: Next-Token Prediction is All You Need**](https://arxiv.org/pdf/2409.18869) <br> | arXiv | 2024-09-27 | [Github](https://github.com/baaivision/Emu3) | Local Demo |
| [**Llama 3.2: Revolutionizing edge AI and vision with open, customizable models**](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) | Meta | 2024-09-25 | - | [Demo](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) | 
| [**Pixtral-12B**](https://mistral.ai/news/pixtral-12b/) | Mistral | 2024-09-17 | - | - |
| ![Star](https://img.shields.io/github/stars/salesforce/LAVIS.svg?style=social&label=Star) <br> [**xGen-MM (BLIP-3): A Family of Open Large Multimodal Models**](https://arxiv.org/pdf/2408.08872) <br> | arXiv | 2024-08-16 | [Github](https://github.com/salesforce/LAVIS/tree/xgen-mm) | - |
| [**The Llama 3 Herd of Models**](https://arxiv.org/pdf/2407.21783) | arXiv | 2024-07-31 | - | - |


## Others
|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/tingyu215/TS-LLaVA.svg?style=social&label=Star) <br> [**TS-LLaVA: Constructing Visual Tokens through Thumbnail-and-Sampling for Training-Free Video Large Language Models**](https://arxiv.org/pdf/2411.11066) <br> | arXiv | 2024-11-17 | [Github](https://github.com/tingyu215/TS-LLaVA) | - |
| ![Star](https://img.shields.io/github/stars/ys-zong/VLGuard.svg?style=social&label=Star) <br> [**Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models**](https://arxiv.org/pdf/2402.02207.pdf) <br> | arXiv | 2024-02-03 | [Github](https://github.com/ys-zong/VLGuard) | - |
| ![Star](https://img.shields.io/github/stars/SHI-Labs/VCoder.svg?style=social&label=Star) <br> [**VCoder: Versatile Vision Encoders for Multimodal Large Language Models**](https://arxiv.org/pdf/2312.14233.pdf) <br> | arXiv | 2023-12-21 | [Github](https://github.com/SHI-Labs/VCoder) | Local Demo | 
| ![Star](https://img.shields.io/github/stars/dvlab-research/Prompt-Highlighter.svg?style=social&label=Star) <br> [**Prompt Highlighter: Interactive Control for Multi-Modal LLMs**](https://arxiv.org/pdf/2312.04302.pdf) <br> | arXiv | 2023-12-07 | [Github](https://github.com/dvlab-research/Prompt-Highlighter) | - |
| ![Star](https://img.shields.io/github/stars/AIlab-CVC/SEED.svg?style=social&label=Star) <br> [**Planting a SEED of Vision in Large Language Model**](https://arxiv.org/pdf/2307.08041.pdf) <br> | arXiv | 2023-07-16 | [Github](https://github.com/AILab-CVC/SEED) |
| ![Star](https://img.shields.io/github/stars/huawei-noah/Efficient-Computing.svg?style=social&label=Star) <br> [**Can Large Pre-trained Models Help Vision Models on Perception Tasks?**](https://arxiv.org/pdf/2306.00693.pdf) <br> | arXiv | 2023-06-01 | [Github](https://github.com/huawei-noah/Efficient-Computing/tree/master/GPT4Image/) | - | 
| ![Star](https://img.shields.io/github/stars/yuhangzang/ContextDET.svg?style=social&label=Star) <br> [**Contextual Object Detection with Multimodal Large Language Models**](https://arxiv.org/pdf/2305.18279.pdf) <br> | arXiv | 2023-05-29 | [Github](https://github.com/yuhangzang/ContextDET) | [Demo](https://huggingface.co/spaces/yuhangzang/ContextDet-Demo) |
| ![Star](https://img.shields.io/github/stars/kohjingyu/gill.svg?style=social&label=Star) <br> [**Generating Images with Multimodal Language Models**](https://arxiv.org/pdf/2305.17216.pdf) <br> | arXiv | 2023-05-26 | [Github](https://github.com/kohjingyu/gill) | - |
| ![Star](https://img.shields.io/github/stars/yunqing-me/AttackVLM.svg?style=social&label=Star) <br> [**On Evaluating Adversarial Robustness of Large Vision-Language Models**](https://arxiv.org/pdf/2305.16934.pdf) <br> | arXiv | 2023-05-26 | [Github](https://github.com/yunqing-me/AttackVLM) | - | 
| ![Star](https://img.shields.io/github/stars/kohjingyu/fromage.svg?style=social&label=Star) <br> [**Grounding Language Models to Images for Multimodal Inputs and Outputs**](https://arxiv.org/pdf/2301.13823.pdf) <br> | ICML | 2023-01-31 | [Github](https://github.com/kohjingyu/fromage) | [Demo](https://huggingface.co/spaces/jykoh/fromage) |

---

# Awesome Datasets

## Datasets of Pre-Training for Alignment
| Name | Paper | Type | Modalities |
|:-----|:-----:|:----:|:----------:|
| **ShareGPT4Video** | [ShareGPT4Video: Improving Video Understanding and Generation with Better Captions](https://arxiv.org/pdf/2406.04325v1) | Caption | Video-Text |
| **COYO-700M** | [COYO-700M: Image-Text Pair Dataset](https://github.com/kakaobrain/coyo-dataset/) | Caption | Image-Text |
| **ShareGPT4V** | [ShareGPT4V: Improving Large Multi-Modal Models with Better Captions](https://arxiv.org/pdf/2311.12793.pdf) | Caption | Image-Text |
| **AS-1B** | [The All-Seeing Project: Towards Panoptic Visual Recognition and Understanding of the Open World](https://arxiv.org/pdf/2308.01907.pdf) | Hybrid | Image-Text |
| **InternVid** | [InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation](https://arxiv.org/pdf/2307.06942.pdf) | Caption | Video-Text |
| **MS-COCO** | [Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312.pdf) | Caption | Image-Text |
| **SBU Captions** | [Im2Text: Describing Images Using 1 Million Captioned Photographs](https://proceedings.neurips.cc/paper/2011/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf) | Caption | Image-Text |
| **Conceptual Captions** | [Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning](https://aclanthology.org/P18-1238.pdf) | Caption | Image-Text |
| **LAION-400M** | [LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs](https://arxiv.org/pdf/2111.02114.pdf) | Caption | Image-Text |
| **VG Captions** |[Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations](https://link.springer.com/content/pdf/10.1007/s11263-016-0981-7.pdf) | Caption | Image-Text |
| **Flickr30k** | [Flickr30k Entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models](https://openaccess.thecvf.com/content_iccv_2015/papers/Plummer_Flickr30k_Entities_Collecting_ICCV_2015_paper.pdf) | Caption | Image-Text |
| **AI-Caps** | [AI Challenger : A Large-scale Dataset for Going Deeper in Image Understanding](https://arxiv.org/pdf/1711.06475.pdf) | Caption | Image-Text | 
| **Wukong Captions** | [Wukong: A 100 Million Large-scale Chinese Cross-modal Pre-training Benchmark](https://proceedings.neurips.cc/paper_files/paper/2022/file/a90b9a09a6ee43d6631cf42e225d73b4-Paper-Datasets_and_Benchmarks.pdf) | Caption | Image-Text |
| **GRIT** | [Kosmos-2: Grounding Multimodal Large Language Models to the World](https://arxiv.org/pdf/2306.14824.pdf) | Caption | Image-Text-Bounding-Box |
| **Youku-mPLUG** | [Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Dataset for Pre-training and Benchmarks](https://arxiv.org/pdf/2306.04362.pdf) | Caption | Video-Text |
| **MSR-VTT** | [MSR-VTT: A Large Video Description Dataset for Bridging Video and Language](https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf) | Caption | Video-Text |
| **Webvid10M** | [Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval](https://arxiv.org/pdf/2104.00650.pdf) | Caption | Video-Text |
| **WavCaps** | [WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research](https://arxiv.org/pdf/2303.17395.pdf) | Caption | Audio-Text |
| **AISHELL-1** | [AISHELL-1: An open-source Mandarin speech corpus and a speech recognition baseline](https://arxiv.org/pdf/1709.05522.pdf) | ASR | Audio-Text |
| **AISHELL-2** | [AISHELL-2: Transforming Mandarin ASR Research Into Industrial Scale](https://arxiv.org/pdf/1808.10583.pdf) | ASR | Audio-Text |
| **VSDial-CN** | [X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages](https://arxiv.org/pdf/2305.04160.pdf) | ASR | Image-Audio-Text |


## Datasets of Multimodal Instruction Tuning
| Name | Paper | Link | Notes |
|:-----|:-----:|:----:|:-----:|
| **Inst-IT Dataset** | [Inst-IT: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning](https://arxiv.org/pdf/2412.03565) | [Link](https://github.com/inst-it/inst-it) | An instruction-tuning dataset which contains fine-grained multi-level annotations for 21k videos and 51k images |
| **E.T. Instruct 164K** | [E.T. Bench: Towards Open-Ended Event-Level Video-Language Understanding](https://arxiv.org/pdf/2409.18111) | [Link](https://github.com/PolyU-ChenLab/ETBench) | An instruction-tuning dataset for time-sensitive video understanding |
| **MSQA** | [Multi-modal Situated Reasoning in 3D Scenes](https://arxiv.org/pdf/2409.02389) | [Link](https://msr3d.github.io/) | A large scale dataset for multi-modal situated reasoning in 3D scenes |
| **MM-Evol** | [MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct](https://arxiv.org/pdf/2409.05840) | [Link](https://mmevol.github.io/home_page.html) | An instruction dataset with rich diversity | 
| **UNK-VQA** | [UNK-VQA: A Dataset and a Probe into the Abstention Ability of Multi-modal Large Models](https://arxiv.org/pdf/2310.10942) | [Link](https://github.com/guoyang9/UNK-VQA) | A dataset designed to teach models to refrain from answering unanswerable questions |
| **VEGA** | [VEGA: Learning Interleaved Image-Text Comprehension in Vision-Language Large Models](https://arxiv.org/pdf/2406.10228) | [Link](https://github.com/zhourax/VEGA) | A dataset for enhancing model capabilities in comprehension of interleaved information | 
| **ALLaVA-4V** | [ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model](https://arxiv.org/pdf/2402.11684.pdf) | [Link](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) | Vision and language caption and instruction dataset generated by GPT4V | 
| **IDK** | [Visually Dehallucinative Instruction Generation: Know What You Don't Know](https://arxiv.org/pdf/2402.09717.pdf) | [Link](https://github.com/ncsoft/idk) | Dehallucinative visual instruction for "I Know" hallucination |
| **CAP2QA** | [Visually Dehallucinative Instruction Generation](https://arxiv.org/pdf/2402.08348.pdf) | [Link](https://github.com/ncsoft/cap2qa) | Image-aligned visual instruction dataset |
| **M3DBench** | [M3DBench: Let's Instruct Large Models with Multi-modal 3D Prompts](https://arxiv.org/pdf/2312.10763.pdf) | [Link](https://github.com/OpenM3D/M3DBench) | A large-scale 3D instruction tuning dataset |
| **ViP-LLaVA-Instruct** | [Making Large Multimodal Models Understand Arbitrary Visual Prompts](https://arxiv.org/pdf/2312.00784.pdf) | [Link](https://huggingface.co/datasets/mucai/ViP-LLaVA-Instruct) |  A mixture of LLaVA-1.5 instruction data and the region-level visual prompting data |
| **LVIS-Instruct4V** | [To See is to Believe: Prompting GPT-4V for Better Visual Instruction Tuning](https://arxiv.org/pdf/2311.07574.pdf) | [Link](https://huggingface.co/datasets/X2FD/LVIS-Instruct4V) | A visual instruction dataset via self-instruction from GPT-4V |
| **ComVint** | [What Makes for Good Visual Instructions? Synthesizing Complex Visual Reasoning Instructions for Visual Instruction Tuning](https://arxiv.org/pdf/2311.01487.pdf) | [Link](https://github.com/RUCAIBox/ComVint#comvint-data) | A synthetic instruction dataset for complex visual reasoning |
| **SparklesDialogue** | [✨Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models](https://arxiv.org/pdf/2308.16463.pdf) | [Link](https://github.com/HYPJUDY/Sparkles#sparklesdialogue) | A machine-generated dialogue dataset tailored for word-level interleaved multi-image and text interactions to augment the conversational competence of instruction-following LLMs across multiple images and dialogue turns. |
| **StableLLaVA** | [StableLLaVA: Enhanced Visual Instruction Tuning with Synthesized Image-Dialogue Data](https://arxiv.org/pdf/2308.10253v1.pdf) | [Link](https://github.com/icoz69/StableLLAVA) | A cheap and effective approach to collect visual instruction tuning data |
| **M-HalDetect** | [Detecting and Preventing Hallucinations in Large Vision Language Models](https://arxiv.org/pdf/2308.06394.pdf) | [Coming soon]() | A dataset used to train and benchmark models for hallucination detection and prevention | 
| **MGVLID** | [ChatSpot: Bootstrapping Multimodal LLMs via Precise Referring Instruction Tuning](https://arxiv.org/pdf/2307.09474.pdf) | - | A high-quality instruction-tuning dataset including image-text and region-text pairs |
| **BuboGPT** | [BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs](https://arxiv.org/pdf/2307.08581.pdf) | [Link](https://huggingface.co/datasets/magicr/BuboGPT) | A high-quality instruction-tuning dataset including audio-text audio caption data and audio-image-text localization data |
| **SVIT** | [SVIT: Scaling up Visual Instruction Tuning](https://arxiv.org/pdf/2307.04087.pdf) | [Link](https://huggingface.co/datasets/BAAI/SVIT) | A large-scale dataset with 4.2M informative visual instruction tuning data, including conversations, detailed descriptions, complex reasoning and referring QAs |
| **mPLUG-DocOwl** | [mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding](https://arxiv.org/pdf/2307.02499.pdf) | [Link](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocLLM) | An instruction tuning dataset featuring a wide range of visual-text understanding tasks including OCR-free document understanding | 
| **PF-1M** | [Visual Instruction Tuning with Polite Flamingo](https://arxiv.org/pdf/2307.01003.pdf) | [Link](https://huggingface.co/datasets/chendelong/PF-1M/tree/main) | A collection of 37 vision-language datasets with responses rewritten by Polite Flamingo. | 
| **ChartLlama** | [ChartLlama: A Multimodal LLM for Chart Understanding and Generation](https://arxiv.org/pdf/2311.16483.pdf) | [Link](https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset) | A multi-modal instruction-tuning dataset for chart understanding and generation |
| **LLaVAR** | [LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding](https://arxiv.org/pdf/2306.17107.pdf) | [Link](https://llavar.github.io/#data) | A visual instruction-tuning dataset for Text-rich Image Understanding | 
| **MotionGPT** | [MotionGPT: Human Motion as a Foreign Language](https://arxiv.org/pdf/2306.14795.pdf) | [Link](https://github.com/OpenMotionLab/MotionGPT) | A instruction-tuning dataset including multiple human motion-related tasks |
| **LRV-Instruction** | [Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning](https://arxiv.org/pdf/2306.14565.pdf) | [Link](https://github.com/FuxiaoLiu/LRV-Instruction#visual-instruction-data-lrv-instruction) | Visual instruction tuning dataset for addressing hallucination issue | 
| **Macaw-LLM** | [Macaw-LLM: Multi-Modal Language Modeling with Image, Audio, Video, and Text Integration](https://arxiv.org/pdf/2306.09093.pdf) | [Link](https://github.com/lyuchenyang/Macaw-LLM/tree/main/data) | A large-scale multi-modal instruction dataset in terms of multi-turn dialogue | 
| **LAMM-Dataset** | [LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark](https://arxiv.org/pdf/2306.06687.pdf) | [Link](https://github.com/OpenLAMM/LAMM#lamm-dataset) | A comprehensive multi-modal instruction tuning dataset |
| **Video-ChatGPT** | [Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models](https://arxiv.org/pdf/2306.05424.pdf) | [Link](https://github.com/mbzuai-oryx/Video-ChatGPT#video-instruction-dataset-open_file_folder) | 100K high-quality video instruction dataset | 
| **MIMIC-IT** | [MIMIC-IT: Multi-Modal In-Context Instruction Tuning](https://arxiv.org/pdf/2306.05425.pdf) | [Link](https://github.com/Luodian/Otter/blob/main/mimic-it/README.md) | Multimodal in-context instruction tuning |
| **M<sup>3</sup>IT** | [M<sup>3</sup>IT: A Large-Scale Dataset towards Multi-Modal Multilingual Instruction Tuning](https://arxiv.org/pdf/2306.04387.pdf) | [Link](https://huggingface.co/datasets/MMInstruction/M3IT) | Large-scale, broad-coverage multimodal instruction tuning dataset | 
| **LLaVA-Med** | [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://arxiv.org/pdf/2306.00890.pdf) | [Coming soon](https://github.com/microsoft/LLaVA-Med#llava-med-dataset) | A large-scale, broad-coverage biomedical instruction-following dataset |
| **GPT4Tools** | [GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction](https://arxiv.org/pdf/2305.18752.pdf) | [Link](https://github.com/StevenGrove/GPT4Tools#dataset) | Tool-related instruction datasets |
| **MULTIS** | [ChatBridge: Bridging Modalities with Large Language Model as a Language Catalyst](https://arxiv.org/pdf/2305.16103.pdf) | [Coming soon](https://iva-chatbridge.github.io/) | Multimodal instruction tuning dataset covering 16 multimodal tasks |
| **DetGPT** | [DetGPT: Detect What You Need via Reasoning](https://arxiv.org/pdf/2305.14167.pdf) | [Link](https://github.com/OptimalScale/DetGPT/tree/main/dataset) |  Instruction-tuning dataset with 5000 images and around 30000 query-answer pairs|
| **PMC-VQA** | [PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering](https://arxiv.org/pdf/2305.10415.pdf) | [Coming soon](https://xiaoman-zhang.github.io/PMC-VQA/) | Large-scale medical visual question-answering dataset |
| **VideoChat** | [VideoChat: Chat-Centric Video Understanding](https://arxiv.org/pdf/2305.06355.pdf) | [Link](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) | Video-centric multimodal instruction dataset |
| **X-LLM** | [X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages](https://arxiv.org/pdf/2305.04160.pdf) | [Link](https://github.com/phellonchen/X-LLM) | Chinese multimodal instruction dataset |
| **LMEye** | [LMEye: An Interactive Perception Network for Large Language Models](https://arxiv.org/pdf/2305.03701.pdf) | [Link](https://huggingface.co/datasets/YunxinLi/Multimodal_Insturction_Data_V2) | A multi-modal instruction-tuning dataset |
| **cc-sbu-align** | [MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models](https://arxiv.org/pdf/2304.10592.pdf) | [Link](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align) | Multimodal aligned dataset for improving model's usability and generation's fluency |
| **LLaVA-Instruct-150K** | [Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485.pdf) | [Link](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | Multimodal instruction-following data generated by GPT|
| **MultiInstruct** | [MultiInstruct: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning](https://arxiv.org/pdf/2212.10773.pdf) | [Link](https://github.com/VT-NLP/MultiInstruct) | The first multimodal instruction tuning benchmark dataset |

## Datasets of In-Context Learning
| Name | Paper | Link | Notes |
|:-----|:-----:|:----:|:-----:|
| **MIC** | [MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning](https://arxiv.org/pdf/2309.07915.pdf) | [Link](https://huggingface.co/datasets/BleachNick/MIC_full) | A manually constructed instruction tuning dataset including interleaved text-image inputs, inter-related multiple image inputs, and multimodal in-context learning inputs. |
| **MIMIC-IT** | [MIMIC-IT: Multi-Modal In-Context Instruction Tuning](https://arxiv.org/pdf/2306.05425.pdf) | [Link](https://github.com/Luodian/Otter/blob/main/mimic-it/README.md) | Multimodal in-context instruction dataset|

## Datasets of Multimodal Chain-of-Thought
| Name | Paper | Link | Notes |
|:-----|:-----:|:----:|:-----:|
| **EMER** | [Explainable Multimodal Emotion Reasoning](https://arxiv.org/pdf/2306.15401.pdf) | [Coming soon](https://github.com/zeroQiaoba/Explainable-Multimodal-Emotion-Reasoning) | A benchmark dataset for explainable emotion reasoning task |
| **EgoCOT** | [EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought](https://arxiv.org/pdf/2305.15021.pdf) | [Coming soon](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch) | Large-scale embodied planning dataset |
| **VIP** | [Let’s Think Frame by Frame: Evaluating Video Chain of Thought with Video Infilling and Prediction](https://arxiv.org/pdf/2305.13903.pdf) | [Coming soon]() | An inference-time dataset that can be used to evaluate VideoCOT |
| **ScienceQA** | [Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering](https://proceedings.neurips.cc/paper_files/paper/2022/file/11332b6b6cf4485b84afadb1352d3a9a-Paper-Conference.pdf) | [Link](https://github.com/lupantech/ScienceQA#ghost-download-the-dataset) | Large-scale multi-choice dataset, featuring multimodal science questions and diverse domains | 

## Datasets of Multimodal RLHF
| Name | Paper | Link | Notes |
|:-----|:-----:|:----:|:-----:|
| **VLFeedback** | [Silkie: Preference Distillation for Large Visual Language Models](https://arxiv.org/pdf/2312.10665.pdf) | [Link](https://huggingface.co/datasets/MMInstruction/VLFeedback) | A vision-language feedback dataset annotated by AI |

## Benchmarks for Evaluation
| Name | Paper | Link | Notes |
|:-----|:-----:|:----:|:-----:|
| **Inst-IT Bench** | [Inst-IT: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning](https://arxiv.org/pdf/2412.03565) | [Link](https://github.com/inst-it/inst-it) | A benchmark to evaluate fine-grained instance-level understanding in images and videos |
| **M<sup>3</sup>CoT** | [M<sup>3</sup>CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Thought](https://arxiv.org/pdf/2405.16473) | [Link](https://github.com/LightChen233/M3CoT) | A multi-domain, multi-step benchmark for multimodal CoT |
| **MMGenBench** | [MMGenBench: Evaluating the Limits of LMMs from the Text-to-Image Generation Perspective](https://arxiv.org/pdf/2411.14062) | [Link](https://github.com/lerogo/MMGenBench) | A benchmark that gauges the performance of constructing image-generation prompt given an image |
| **MiCEval** | [MiCEval: Unveiling Multimodal Chain of Thought's Quality via Image Description and Reasoning Steps](https://arxiv.org/pdf/2410.14668) | [Link](https://github.com/alenai97/MiCEval) | A multimodal CoT benchmark to evaluate MLLMs' reasoning capabilities |
| **LiveXiv** | [LiveXiv -- A Multi-Modal Live Benchmark Based on Arxiv Papers Content](https://arxiv.org/pdf/2410.10783) | [Link](https://huggingface.co/datasets/LiveXiv/LiveXiv) | A live benchmark based on arXiv papers |
| **TemporalBench** | [TemporalBench: Benchmarking Fine-grained Temporal Understanding for Multimodal Video Models](https://arxiv.org/pdf/2410.10818) | [Link](https://huggingface.co/datasets/microsoft/TemporalBench) | A benchmark for evaluation of fine-grained temporal understanding |
| **OmniBench** | [OmniBench: Towards The Future of Universal Omni-Language Models](https://arxiv.org/pdf/2409.15272) | [Link](https://huggingface.co/datasets/m-a-p/OmniBench) | A benchmark that evaluates models' capabilities of processing visual, acoustic, and textual inputs simultaneously |
| **MME-RealWorld** | [MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans?](https://arxiv.org/pdf/2408.13257) | [Link](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld) | A challenging benchmark that involves real-life scenarios |
| **VELOCITI** | [VELOCITI: Can Video-Language Models Bind Semantic Concepts through Time?](https://arxiv.org/pdf/2406.10889) | [Link](https://github.com/katha-ai/VELOCITI) | A video benhcmark that evaluates on perception and binding capabilities |
| **MMR** | [Seeing Clearly, Answering Incorrectly: A Multimodal Robustness Benchmark for Evaluating MLLMs on Leading Questions](https://arxiv.org/pdf/2406.10638.pdf) | [Link](https://github.com/BAAI-DCAI/Multimodal-Robustness-Benchmark) | A benchmark for measuring MLLMs' understanding capability and robustness to leading questions |
| **CharXiv** | [CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs](https://arxiv.org/pdf/2406.18521) | [Link](https://huggingface.co/datasets/princeton-nlp/CharXiv) | Chart understanding benchmark curated by human experts |
| **Video-MME** | [Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis](https://arxiv.org/pdf/2405.21075) | [Link](https://github.com/BradyFU/Video-MME) | A comprehensive evaluation benchmark of Multi-modal LLMs in video analysis |
| **VL-ICL Bench** | [VL-ICL Bench: The Devil in the Details of Benchmarking Multimodal In-Context Learning](https://arxiv.org/pdf/2403.13164.pdf) | [Link](https://github.com/ys-zong/VL-ICL) | A benchmark for M-ICL evaluation, covering a wide spectrum of tasks |
| **TempCompass** | [TempCompass: Do Video LLMs Really Understand Videos?](https://arxiv.org/pdf/2403.00476.pdf) | [Link](https://github.com/llyx97/TempCompass) | A benchmark to evaluate the temporal perception ability of Video LLMs |
| **GVLQA** | [GITA: Graph to Visual and Textual Integration for Vision-Language Graph Reasoning](https://arxiv.org/pdf/2402.02130) | [Link](https://huggingface.co/collections/Yanbin99/gvlqa-datasets-65c705c9488606617e246bd3) | A benchmark for evaluation of graph reasoning capabilities |
| **CoBSAT** | [Can MLLMs Perform Text-to-Image In-Context Learning?](https://arxiv.org/pdf/2402.01293.pdf) | [Link](https://huggingface.co/datasets/yzeng58/CoBSAT) | A benchmark for text-to-image ICL |
| **VQAv2-IDK** | [Visually Dehallucinative Instruction Generation: Know What You Don't Know](https://arxiv.org/pdf/2402.09717.pdf) | [Link](https://github.com/ncsoft/idk) | A benchmark for assessing "I Know" visual hallucination |
| **Math-Vision** | [Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset](https://arxiv.org/pdf/2402.14804.pdf) | [Link](https://github.com/mathvision-cuhk/MathVision) | A diverse mathematical reasoning benchmark |
| **SciMMIR** | [SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval](https://arxiv.org/pdf/2401.13478) | [Link](https://github.com/Wusiwei0410/SciMMIR) | | A benchmark for multi-modal information retrieval in the science domain |
| **CMMMU** | [CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark](https://arxiv.org/pdf/2401.11944.pdf) | [Link](https://github.com/CMMMU-Benchmark/CMMMU) | A Chinese benchmark involving reasoning and knowledge across multiple disciplines |
| **MMCBench** | [Benchmarking Large Multimodal Models against Common Corruptions](https://arxiv.org/pdf/2401.11943.pdf) | [Link](https://github.com/sail-sg/MMCBench) | A benchmark for examining self-consistency under common corruptions |
| **MMVP** | [Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](https://arxiv.org/pdf/2401.06209.pdf) | [Link](https://github.com/tsb0601/MMVP) | A benchmark for assessing visual capabilities |
| **TimeIT** | [TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding](https://arxiv.org/pdf/2312.02051.pdf) | [Link](https://huggingface.co/datasets/ShuhuaiRen/TimeIT) | A video instruction-tuning dataset with timestamp annotations, covering diverse time-sensitive video-understanding tasks. |
| **ViP-Bench** | [Making Large Multimodal Models Understand Arbitrary Visual Prompts](https://arxiv.org/pdf/2312.00784.pdf) | [Link](https://huggingface.co/datasets/mucai/ViP-Bench) | A benchmark for visual prompts |
| **M3DBench** | [M3DBench: Let's Instruct Large Models with Multi-modal 3D Prompts](https://arxiv.org/pdf/2312.10763.pdf) | [Link](https://github.com/OpenM3D/M3DBench) | A 3D-centric benchmark |
| **Video-Bench** | [Video-Bench: A Comprehensive Benchmark and Toolkit for Evaluating Video-based Large Language Models](https://arxiv.org/pdf/2311.16103.pdf) | [Link](https://github.com/PKU-YuanGroup/Video-Bench) | A benchmark for video-MLLM evaluation |
| **Charting-New-Territories** | [Charting New Territories: Exploring the Geographic and Geospatial Capabilities of Multimodal LLMs](https://arxiv.org/pdf/2311.14656.pdf) | [Link](https://github.com/jonathan-roberts1/charting-new-territories) | A benchmark for evaluating geographic and geospatial capabilities |
| **MLLM-Bench** | [MLLM-Bench, Evaluating Multi-modal LLMs using GPT-4V](https://arxiv.org/pdf/2311.13951.pdf) | [Link](https://github.com/FreedomIntelligence/MLLM-Bench) | GPT-4V evaluation with per-sample criteria |
| **BenchLMM** | [BenchLMM: Benchmarking Cross-style Visual Capability of Large Multimodal Models](https://arxiv.org/pdf/2312.02896.pdf) | [Link](https://huggingface.co/datasets/AIFEG/BenchLMM) | A benchmark for assessment of the robustness against different image styles |
| **MMC-Benchmark** | [MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning](https://arxiv.org/pdf/2311.10774.pdf) | [Link](https://github.com/FuxiaoLiu/MMC) | A comprehensive human-annotated benchmark with distinct tasks evaluating reasoning capabilities over charts |
| **MVBench** | [MVBench: A Comprehensive Multi-modal Video Understanding Benchmark](https://arxiv.org/pdf/2311.17005.pdf) | [Link](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md) | A comprehensive multimodal benchmark for video understanding |
| **Bingo** | [Holistic Analysis of Hallucination in GPT-4V(ision): Bias and Interference Challenges](https://arxiv.org/pdf/2311.03287.pdf) | [Link](https://github.com/gzcch/Bingo) | A benchmark for hallucination evaluation that focuses on two common types |
| **MagnifierBench** | [OtterHD: A High-Resolution Multi-modality Model](https://arxiv.org/pdf/2311.04219.pdf) | [Link](https://huggingface.co/datasets/Otter-AI/MagnifierBench) | A benchmark designed to probe models' ability of fine-grained perception |
| **HallusionBench** | [HallusionBench: You See What You Think? Or You Think What You See? An Image-Context Reasoning Benchmark Challenging for GPT-4V(ision), LLaVA-1.5, and Other Multi-modality Models](https://arxiv.org/pdf/2310.14566.pdf) | [Link](https://github.com/tianyi-lab/HallusionBench) |An image-context reasoning benchmark for evaluation of hallucination |
| **PCA-EVAL** | [Towards End-to-End Embodied Decision Making via Multi-modal Large Language Model: Explorations with GPT4-Vision and Beyond](https://arxiv.org/pdf/2310.02071.pdf) | [Link](https://github.com/pkunlp-icler/PCA-EVAL) | A benchmark for evaluating multi-domain embodied decision-making. |
| **MMHal-Bench** | [Aligning Large Multimodal Models with Factually Augmented RLHF](https://arxiv.org/pdf/2309.14525.pdf) | [Link](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench) | A benchmark for hallucination evaluation |
| **MathVista** | [MathVista: Evaluating Math Reasoning in Visual Contexts with GPT-4V, Bard, and Other Large Multimodal Models](https://arxiv.org/pdf/2310.02255.pdf) | [Link](https://huggingface.co/datasets/AI4Math/MathVista) | A benchmark that challenges both visual and math reasoning capabilities |
| **SparklesEval** | [✨Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models](https://arxiv.org/pdf/2308.16463.pdf) | [Link](https://github.com/HYPJUDY/Sparkles#sparkleseval) | A GPT-assisted benchmark for quantitatively assessing a model's conversational competence across multiple images and dialogue turns based on three distinct criteria. |
| **ISEKAI** | [Link-Context Learning for Multimodal LLMs](https://arxiv.org/pdf/2308.07891.pdf) | [Link](https://huggingface.co/ISEKAI-Portal) | A benchmark comprising exclusively of unseen generated image-label pairs designed for link-context learning |
| **M-HalDetect** | [Detecting and Preventing Hallucinations in Large Vision Language Models](https://arxiv.org/pdf/2308.06394.pdf) | [Coming soon]() | A dataset used to train and benchmark models for hallucination detection and prevention | 
| **I4** | [Empowering Vision-Language Models to Follow Interleaved Vision-Language Instructions](https://arxiv.org/pdf/2308.04152.pdf) | [Link](https://github.com/DCDmllm/Cheetah) | A benchmark to comprehensively evaluate the instruction following ability on complicated interleaved vision-language instructions | 
| **SciGraphQA** | [SciGraphQA: A Large-Scale Synthetic Multi-Turn Question-Answering Dataset for Scientific Graphs](https://arxiv.org/pdf/2308.03349.pdf) | [Link](https://github.com/findalexli/SciGraphQA#data) | A large-scale chart-visual question-answering dataset |
| **MM-Vet**| [MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities](https://arxiv.org/pdf/2308.02490.pdf) | [Link](https://github.com/yuweihao/MM-Vet) | An evaluation benchmark that examines large multimodal models on complicated multimodal tasks |
| **SEED-Bench** | [SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension](https://arxiv.org/pdf/2307.16125.pdf) | [Link](https://github.com/AILab-CVC/SEED-Bench) | A benchmark for evaluation of generative comprehension in MLLMs | 
| **MMBench** | [MMBench: Is Your Multi-modal Model an All-around Player?](https://arxiv.org/pdf/2307.06281.pdf) | [Link](https://github.com/open-compass/MMBench) | A systematically-designed objective benchmark for robustly evaluating the various abilities of vision-language models|
| **Lynx** | [What Matters in Training a GPT4-Style Language Model with Multimodal Inputs?](https://arxiv.org/pdf/2307.02469.pdf) | [Link](https://github.com/bytedance/lynx-llm#prepare-data) |  A comprehensive evaluation benchmark including both image and video tasks |
| **GAVIE** | [Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning](https://arxiv.org/pdf/2306.14565.pdf) | [Link](https://github.com/FuxiaoLiu/LRV-Instruction#evaluationgavie) | A benchmark to evaluate the hallucination and instruction following ability | 
| **MME** | [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://arxiv.org/pdf/2306.13394.pdf) | [Link](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | A comprehensive MLLM Evaluation benchmark |
| **LVLM-eHub** | [LVLM-eHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models](https://arxiv.org/pdf/2306.09265.pdf) | [Link](https://github.com/OpenGVLab/Multi-Modality-Arena) | An evaluation platform for MLLMs |
| **LAMM-Benchmark** | [LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark](https://arxiv.org/pdf/2306.06687.pdf) | [Link](https://github.com/OpenLAMM/LAMM#lamm-benchmark) | A benchmark for evaluating  the quantitative performance of MLLMs on various2D/3D vision tasks |
| **M3Exam** | [M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models](https://arxiv.org/pdf/2306.05179.pdf) | [Link](https://github.com/DAMO-NLP-SG/M3Exam) |  A multilingual, multimodal, multilevel benchmark for evaluating MLLM |
| **OwlEval** | [mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality](https://arxiv.org/pdf/2304.14178.pdf) | [Link](https://github.com/X-PLUG/mPLUG-Owl/tree/main/OwlEval) | Dataset for evaluation on multiple capabilities |

## Others
| Name | Paper | Link | Notes |
|:-----|:-----:|:----:|:-----:|
| **IMAD** | [IMAD: IMage-Augmented multi-modal Dialogue](https://arxiv.org/pdf/2305.10512.pdf) | [Link](https://github.com/VityaVitalich/IMAD) | Multimodal dialogue dataset|
| **Video-ChatGPT** | [Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models](https://arxiv.org/pdf/2306.05424.pdf) | [Link](https://github.com/mbzuai-oryx/Video-ChatGPT#quantitative-evaluation-bar_chart) | A quantitative evaluation framework for video-based dialogue models |
| **CLEVR-ATVC** | [Accountable Textual-Visual Chat Learns to Reject Human Instructions in Image Re-creation](https://arxiv.org/pdf/2303.05983.pdf) | [Link](https://drive.google.com/drive/folders/1TqBzkyqxOSg1hgCXF8JjpYIAuRV-uVft) | A synthetic multimodal fine-tuning dataset for learning to reject instructions |
| **Fruit-ATVC** | [Accountable Textual-Visual Chat Learns to Reject Human Instructions in Image Re-creation](https://arxiv.org/pdf/2303.05983.pdf) | [Link](https://drive.google.com/drive/folders/1Saaia2rRRb1nz5sKdmpzYdS4jHiMDaP0) | A manually pictured multimodal fine-tuning dataset for learning to reject instructions |
| **InfoSeek** | [Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions?](https://arxiv.org/pdf/2302.11713.pdf) | [Link](https://open-vision-language.github.io/infoseek/) | A VQA dataset that focuses on asking information-seeking questions |
| **OVEN** | [Open-domain Visual Entity Recognition: Towards Recognizing Millions of Wikipedia Entities](https://arxiv.org/pdf/2302.11154.pdf) | [Link](https://open-vision-language.github.io/oven/) | A dataset that focuses on recognizing the Visual Entity on the Wikipedia, from images in the wild |
