# Multi-Level Decoupling and Coordination of Text-Guided Visual Representation for UAV-Based Pine Wilt Disease Detection
![overview](statics/fig1.png)
## :evergreen_tree: Overview

- **Research Background**：Pine Wilt Disease (PWD) poses a devastating threat to global forest ecological security. Accurate detection of infected trees is critical, yet pure visual methods often confuse diseased trees with visually similar objects. Vision-language models can introduce helpful textual prior knowledge, but they generally suffer from interference by task-irrelevant redundant textual information, leading to misaligned cross-modal features. The fundamental challenge is twofold: how to suppress cross-modal redundancy and how to prevent the attenuation of early weak features that are crucial for identifying subtle disease symptoms. 
- **Technical route**：First, pure visual detection confuses infected trees with similar-looking objects. To resolve such confusion, text modality is introduced to provide discriminative semantics. Simultaneously, a global compression strategy removes task-irrelevant redundancy, yielding global semantic guidance fused with text. Second, early weak lesion features are easily lost as the network deepens. To counteract that attenuation, global semantic guidance is leveraged to decouple feature learning, collaboratively strengthening features from two aspects: mining fine-grained details and reactivating subtle early lesion cues. Finally, multi-scale fusion risks re-attenuating the strengthened features. A bidirectional detail-semantic aggregation is therefore applied to preserve these features, thus closing the loop.
- **Core innovation**：
  - Task-specific dataset with aligned cross-modal annotations
  - Coarse-fine dual-granularity labeling framework for imbalanced small targets
  - Text-driven semantically coupled segmentation network (CF-SCSNet)
 
## :card_file_box:Datasets
<div align="center">
    <img src="statics/dataset.png" width="80%">
</div>
  
## :bar_chart: Model test dataset
| **Dataset**         | Dataset download |
| :------------------ | :--------------------- |
| **self-built PWD** |  |
| **Public generalization(FramSeg-M)** | [dataset](https://www.selectdataset.com/dataset/b6bd538e3e21259cf6958130ef5ed70a)   |
| **Public generalization(FramSeg-L)** | [dataset](https://www.selectdataset.com/dataset/b6bd538e3e21259cf6958130ef5ed70a)   |
| **Public generalization(RRSIS-M)** | [dataset](https://www.selectdataset.com/dataset/5bbd39a0d71020dd530930d9bb39eae7)   |
| **Public generalization(RRSIS-L)** | [dataset](https://www.selectdataset.com/dataset/5bbd39a0d71020dd530930d9bb39eae7)   |

## :fallen_leaf: Visualization
<details open>
  <summary>self-built PWD</summary>
  <div align="center">
    <img src="statics/visualization.png" width="80%">
  </div>
</details>

## :computer: Installation
<details open>
  <summary>Dependency installation steps</summary>
  
  1. **Clone this project and create a conda environment:**
     ```bash
     git clone https://github.com/TechJots-Liu/CF-SCSNet.git
     cd CF-SCSNet
     
     conda create -n cf_scsnet python=3.10.9
     conda activate cf_scsnet
  2. **Install pytorch and torchvision matching your CUDA version:**
     ```bash
     pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  3. **Install requirements:**
     ```bash
     pip install -r requirements.txt
  4. **Load the Roberta word encoder locally**
     [Roberta-base](https://huggingface.co/FacebookAI/roberta-base)


