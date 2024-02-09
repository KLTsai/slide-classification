# Offer Slide Classification

## Introduction

This project leverages a machine learning solution to automatically classify presentation slides (.pptx) by analyzing both text and images, aiming to **verify results in organizations from searching** and **reduce manual labeling**.

## Objectives

- Integrate textual and visual information for slide classification.
- Evaluate and compare baseline and advanced machine learning models.
- Conclude optimal parameters combination for baseline and advanced approaches.
- Integrated CI/CD pipeline  into an application

## Key Tool & Libraries Used

- [`python_pptx`](https://python-pptx.readthedocs.io/en/latest/) for handling presentation slides (.pptx)
- [`sklearn`](https://github.com/scikit-learn/scikit-learn) for feature extraction and selection, baseline classifier, and model evaluation, etc.
- Used text dection algorithms by [`mmocr`](https://github.com/open-mmlab/mmocr) for visual preprocessing.
- [`torch`](https://github.com/pytorch/pytorch) for machine learning model development.
- 🤗 [`transformers`](https://huggingface.co/models) for pre-trained model leveraging.

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


## Methodology

- **Slide Preprocessing**: Techniques used to prepare slides for classification.
- **Model Development**: Description of baseline and advanced approaches, including text-based, vision-based, and multi-modal models.
- **Performance Metrics**: Criteria for evaluating model performance.

## Experiments and Results

Overview of experimental setup, model training, and evaluation results. Highlights the superiority of multi-modal information processing in slide classification tasks.

![alt text](code/output/others/all_models_perf_cluster.png)

## How to Use

Instructions on how to set up the environment, train the models, and classify new slides.

## Contributing

Guidelines for contributing to the project, including how to submit issues and pull requests.

## License

Details of the project's license.

## Acknowledgments

Gratitude expressed towards advisors, contributors, and any supporting institutions.
