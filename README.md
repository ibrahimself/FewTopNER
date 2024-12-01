# FewTopNER

**Few-shot learning for joint Named Entity Recognition (NER) and Topic Modeling.**

---

## Overview

**FewTopNER** is a cross-lingual model that integrates **Named Entity Recognition (NER)** and **Topic Modeling** using **Few-shot Learning**. It leverages the **WikiNEuRal dataset** for NER and Wikipedia data for Topic Modeling in five languages:  
- **English**  
- **French**  
- **German**  
- **Spanish**  
- **Italian**

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/ibrahimself/fewtopner.git
cd fewtopner
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Language Models
Download the necessary **spaCy** models for each language:
```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download es_core_news_sm
python -m spacy download it_core_news_sm
```

---

## Data Preparation

### Step 1: Download WikiNEuRal Dataset
Create a directory and download the dataset:
```bash
mkdir -p data/ner/wikineural
# Download from https://github.com/Babelscape/wikineural
```

### Step 2: Download Wikipedia Data
Create a directory and download language-specific Wikipedia dumps:
```bash
mkdir -p data/topic/wikipedia
# Refer to https://dumps.wikimedia.org/ for instructions
```

---

## Usage

### Step 1: Configure Your Experiment
Copy and customize the configuration file:
```bash
cp configs/fewtopner_config.yaml configs/my_experiment.yaml
# Modify `my_experiment.yaml` as needed
```

### Step 2: Train the Model
Run the training script:
```bash
python main.py --config configs/my_experiment.yaml
```

### Step 3: Evaluate a Checkpoint
Evaluate the model using a saved checkpoint:
```bash
python main.py --config configs/my_experiment.yaml --checkpoint outputs/checkpoints/best_model.pt
```

---

## Project Structure

```
fewtopner/
├── configs/                  # Configuration files
│   └── fewtopner_config.yaml
├── data/                     # Data directories
│   ├── ner/
│   │   └── wikineural/
│   └── topic/
│       └── wikipedia/
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   │   ├── preprocessing/
│   │   │   ├── ner_processor.py
│   │   │   └── topic_processor.py
│   │   ├── dataset.py
│   │   └── dataloader.py
│   ├── model/                # Model architecture components
│   │   ├── shared_encoder.py
│   │   ├── entity_branch.py
│   │   ├── topic_branch.py
│   │   ├── bridge.py
│   │   └── fewtopner.py
│   ├── training/             # Training-related scripts
│   │   ├── trainer.py
│   │   ├── episode_builder.py
│   │   └── metrics.py
│   └── utils/                # Utility scripts
│       ├── config.py
│       └── multilingual.py
├── main.py                   # Entry point for training/evaluation
└── requirements.txt          # Python dependencies
```

---

## Docker Support

Easily set up the project using Docker:

### Dockerfile
```dockerfile
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_sm \
    && python -m spacy download fr_core_news_sm \
    && python -m spacy download de_core_news_sm \
    && python -m spacy download es_core_news_sm \
    && python -m spacy download it_core_news_sm

# Set default command
CMD ["python", "main.py", "--config", "configs/fewtopner_config.yaml"]
```

### Build and Run
Build the Docker image and run the container:
```bash
docker build -t fewtopner .
docker run -it --rm fewtopner
```

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)


## Citation

If you use **FewTopNER** in your research, please cite it as follows:
```bibtex
@article{fewtopner2024,
  title={FewTopNER: Few-shot Learning for Joint Named Entity Recognition and Topic Modeling},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```
