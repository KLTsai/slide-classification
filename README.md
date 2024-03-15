# Offer Slide Classification

## Introduction

This project leverages a machine learning solution to automatically classify presentation slides (.pptx) by analyzing both text and images, aiming to **verify results in organizations from searching** and **reduce manual labeling**.

## Objectives

- Integrate textual and visual information for slide classification.
- Evaluate and compare baseline and advanced machine learning models.
- Conclude optimal parameters combination for baseline and advanced approaches.
- Established docker script for CI/CD pipeline.

## Key Tool & Libraries Used

- [`python_pptx`](https://python-pptx.readthedocs.io/en/latest/) for handling presentation slides (.pptx)
- [`plotly`](https://github.com/plotly/plotly.py) & [`seaborn`](https://seaborn.pydata.org/tutorial.html) for exploratory data analysis.
- [`NLTK`](https://www.nltk.org/) for nature language processing.
- [`sklearn`](https://github.com/scikit-learn/scikit-learn) for feature extraction and selection, baseline classifier, and model evaluation, etc.
- Used text dection algorithms by [`mmocr`](https://github.com/open-mmlab/mmocr) for visual preprocessing.
- [`torch`](https://github.com/pytorch/pytorch) for deep learning model development.
- 🤗 [`Transformers`](https://huggingface.co/models) for pre-trained model leveraging.

## Dataset

The dataset for analyzing the offer reference contains 26,485 slide images (size around **6 GB**). Approximately 14% of the total images were comprised of 3,763 images divided into **12 categories**. Samples were selected using stratified random sampling to ensure it was heterogeneous and representative of the entire dataset, considering attributes such as slide categories.

| Category | Index Number | Number of Samples |
| :-- | --: |--:|
| Working Package Examples  | 10 | 668 |
| Working Package Description | 9| 564|
| Reference Details|6|407|
| Consultant Profile|1|370|
|Offer Title|4|365|
|Project Calculation|5|304|
|Initial & Target Situation|2|225|
|Working Package Overview|7|215|
|Compentencies|0|189|
|Target Situation|8|103|
|Initial Situation|3|86|

> **_NOTE:_** This offer reference dataset is considered a confidential asset and is not available for public use at this time.

## Techniques Covered

- **Feature Processing For Baseline Approach**: Three modalities to obtain rich information from presentation slides are **text**, **vision**, and **multi-modal**.

| Approach  | Modalities  | Feature Extraction               | Feature Selection |
|-----------|-------------|----------------------------------|-------------------|
| baseline | Text        | bag of words, TF-IDF             | chi-square test   |
| baseline  | Vision      | shape types as feature vector    | chi-square test   |
| baseline  | Multi-modal | feature union (Text & Vision)    | chi-square test   |

- **Model Development**: Leveraging advanced approaches with Transformers, including text-based, vision-based, and multi-modal models.

| Model Name | Modality | Sequence Manipulation |Pretrained Model|
| :-- | :-- |:--|:--|
|BERT |Text| split sequence into pieces for long texts|[`bert-base-multilingual-cased`](https://huggingface.co/google-bert/bert-base-multilingual-cased)|
|ViT| Vision| geometry processing|[`vit-large-patch16-384`](https://huggingface.co/google/vit-large-patch16-384)|
|LiLT|Multi-modal| bi-directional attention complementation mechanism|[`lilt-xlm-roberta-base`](https://huggingface.co/nielsr/lilt-xlm-roberta-base)|

- **Performance Metrics**: Criteria for evaluating model performance such as F1-Score, Precision, Recall, Average Latency.

## Experiments and Results

Overview of experimental setup, model fine-tuning, and evaluation results. Highlights the superiority of multi-modal information processing in slide classification tasks.

### Multi-Class Classification Results
Our advanced approach (Adv) utilizes three Transformer-based models with each distinct modality. We can assess the variance between predicted and actual class distribution using cross-entropy. To analyze the training learning curves across various modalities, we divide the results based on sequence length or the number of channels. Then, we only select the combinations of hyperparameters with the three lowest validation losses and consider them as the best candidates for optimal parameter configuration. There are different combinations of hyperparameters for advanced approaches for each modality comprising epochs (e), batch size (b), sequence lengths (s), learning rate (lr), grayscale (mono), and RGB(rgb) (see legend labels in [below scatter plot](#accuracy-latency-in-classification-models)). Finally, we choose the optimal combinations of hyperparameters for each modality by comparing validation losses and classification performance from the test set in the analysis of classification results.

 | Transformer Model (sequence length / channel) | Batch | Learning Rate | Epoch | Validation Accuracy | Validation Loss | Test Accuracy | F1-Score (marco) |
|------------------------------------------------|-------|---------------|-------|---------------------|-----------------|---------------|------------------|
 | **BERT_128**                                       | **16**    | **1.3e-05**       | **5**     | **0.9173**              | **0.3275**          | **0.8969**        |**0.8847**           |
| BERT_128                                       | 16    | 4.0e-05       | 3     | 0.8915              | 0.4840          | 0.9072        | 0.8902           |
| BERT_128                                       | 16    | 5.0e-05       | 6     | 0.8837              | 0.5498          | 0.8943        | 0.8880           |
| BERT_256                                       | 16    | 1.3e-05       | 5     | 0.9193              | 0.3036          | 0.9098        | 0.8741           |
 | BERT_256                                       | 16    | 4.0e-05       | 6     | 0.9251              | 0.3507          | 0.8686        | 0.8640           |
 | BERT_256                                       | 16    | 5.0e-05       | 6     | 0.9267              | 0.3553          | 0.8582        | 0.8472           |
 | BERT_512                                       | 16    | 1.3e-05       | 6     | 0.9380              | 0.2738          | 0.8943        | 0.8805           |
| BERT_512                                       | 16    | 4.0e-05       | 6     | 0.9328              | 0.3020          | 0.8840        | 0.8625           |
| BERT_512                                       | 16    | 5.0e-05       | 4     | 0.9251              | 0.3378          | 0.8814        | 0.8735           |
 | ViT_mono                                       | 4     | 1.4e-04       | 6     | 0.8271              | 0.5961          | 0.7851        | 0.6599           |
 | ViT_mono                                       | 4     | 2.5e-04       | 9     | 0.7394              | 0.7981          | 0.7533        | 0.6440           |
 | ViT_mono                                       | 4     | 2.5e-05       | 6     | 0.9415              | 0.2724          | 0.9178        | 0.8680           |
 | **ViT_rgb**                                        | **4**     | **1.4e-04**       | **6**     | **0.9628**              | **0.2031**          | **0.9523**        | **0.9300**           |
  | ViT_rgb                                        | 4     | 2.5e-04       | 8     | 0.9229              | 0.3833          | 0.9098        | 0.8864           |
 | ViT_rgb                                        | 4     | 2.5e-05       | 5     | 0.9441              | 0.3045          | 0.939         | 0.9196           |
| LiLT_128                                       | 8     | 2.0e-06       | 6     | 0.9311              | 0.4233          | 0.9284        | 0.9231           |
  | LiLT_128                                       | 8     | 4.0e-06       | 6     | 0.9521              | 0.2401          |  0.9469        | 0.9516          |
 | LiLT_128                                       | 8     | 6.0e-06       | 6     | 0.9521              | 0.2557          | 0.9549        | 0.9614           |
  | LiLT_256                                       | 8     | 2.0e-06       | 6     | 0.9335              | 0.4066          | 0.9231        | 0.9227           |
  | LiLT_256                                       | 8     | 4.0e-06       | 6     | 0.9548              | 0.2301          | 0.9549        | 0.9606           |
  | LiLT_256                                       | 8     | 6.0e-06       | 6     | 0.9601              | 0.2267          | 0.9655        | 0.9718           |
  | LiLT_512                                       | 8     | 4.0e-05       | 5     | 0.8936              | 0.2397          | 0.9151        | 0.9599           |
  | LiLT_512                                       | 8     | 2.0e-06       | 6     | 0.9335              | 0.3974          | 0.9601        | 0.8997           |
   | **LiLT_512**                                       | **8**     | **6.0e-06**       | **6**     | **0.9441**              | **0.2124**          | **0.9602**        | **0.9657**           |

> **_NOTE:_** Given entries with bold indicate optimal parameters are selected for each modality.


### Accuracy-Latency in Classification Models

Established *warmup* and *timed run iterations* at 10 and 100 times, respectively. We determine the average and standard deviation of latency to serve as a standard for assessing model inference time to make a plot with its accuracy.

![alt text](code/output/others/all_models_perf_cluster.png)

> **_NOTE:_** Dash circles represent optimal parameter combinations in each modality.

## How to Use

- Before running the code, you sholud set up the enviroment we needed by entering the following command into the terminal:

``` terminal
pip install requirements.txt
```
then verify configuration of parameters in `advance_slide_modeling.ipynb`, `advanced_slide_modeling_visual.ipynb`, `advanced_slide_modeling_multimodal.ipynb`

- Please refer to the following parameter table for fine-tuning Transformer models:

| Models            | Sequence Length | Batch | Learning Rate | Epoch |
|-------------------|-----------------|-------|---------------|-------|
| BERT_Seq+Dense    | 128             | 16    | 1.3E-05       | 5     |
| BERT_Seq+Dense    | 128             | 16    | 4.0E-05       | 3     |
| BERT_Seq+Dense    | 128             | 16    | 5.0E-05       | 6     |
| BERT_Seq+Dense    | 256             | 16    | 1.3E-05       | 5     |
| BERT_Seq+Dense    | 256             | 16    | 4.0E-05       | 6     |
| BERT_Seq+Dense    | 256             | 16    | 5.0E-05       | 6     |
| BERT_Seq+Dense    | 512             | 16    | 1.3E-05       | 6     |
| BERT_Seq+Dense    | 512             | 16    | 4.0E-05       | 6     |
| BERT_Seq+Dense    | 512             | 16    | 5.0E-05       | 4     |
| ViT_Mono+Dense    | :x:             | 4     | 1.4E-04       | 6     |
| ViT_Mono+Dense    | :x:             | 4     | 2.5E-04       | 9     |
| ViT_Mono+Dense    | :x:             | 4     | 2.5E-05       | 6     |
| ViT_Color+Dense   | :x:             | 4     | 1.4E-04       | 6     |
| ViT_Color+Dense   | :x:             | 4     | 2.5E-04       | 9     |
| ViT_Color+Dense   | :x:             | 4     | 2.5E-05       | 6     |
| LiLT_Seq+Dense    | 128             | 8    | 2.0E-06       | 6     |
| LILT_Seq+Dense    | 128             | 8    | 4.0E-06       | 6     |
| LILT_Seq+Dense    | 128             | 8    | 6.0E-06       | 6     |
| LiLT_Seq+Dense    | 256             | 8    | 2.0E-06       | 6     |
| LILT_Seq+Dense    | 256             | 8    | 4.0E-06       | 6     |
| LILT_Seq+Dense    | 256             | 8    | 6.0E-06       | 6     |
| LiLT_Seq+Dense    | 512             | 8    | 4.0E-05       | 5     |
| LILT_Seq+Dense    | 512             | 8    | 2.0E-06       | 6     |
| LILT_Seq+Dense    | 512             | 8    | 6.0E-06       | 6     |

> **_NOTE:_** Since ViT model allows each image is split into a sequence of fixed-size non-overlapping patches, it does not provide a sequence length adjustment parameter.

Following instructions on the table of contents in notebooks then obtain results of fine-tuning, users can select an ideal fine-tuned model for deployment after evaluations.
