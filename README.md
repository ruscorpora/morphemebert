# MorphemeBERT: BERT-based Morpheme Segmentation

This repository contains the code for fine-tuning and running inference with BERT-like models for the task of surface morpheme segmentation.

## 1. Setup

### Installation

It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### Project Structure

Your project should be organized as follows for the script to run correctly:

```
.
├── morphbert.py          # The main class file
├── run_experiment.py     # An example script to run training/prediction
├── models.json           # Configuration for models
├── datasets.json         # Configuration for datasets
├── requirements.txt      # Project dependencies
|
├── data/
│   └── your_corpus_name/
│       └── split-by-lemma/
│           ├── 0/
│           │   ├── train.txt
│           │   └── test.txt
│           └── 1/
│               └── ...
│
├── outputs/              # (created automatically) Stores trained model weights
└── results/              # (created automatically) Stores predictions and metrics
```

## 2. Data Format

The input data for training and testing must be a `.txt` file where each line contains a word and its morphemic parsing, separated by a tab.

**Format:** `word<TAB>morpheme1:TYPE1/morpheme2:TYPE2/...`

**Morpheme Types:** `ROOT`, `PREF`, `SUFF`, `END`, `POST`, `LINK`, `HYPN`.

### Example `train.txt` file:

```
переход	пере:PREF/ход:ROOT
безвкусный	без:PREF/вкус:ROOT/н:SUFF/ый:END
слово	слов:ROOT/о:END
кто-то	кто:ROOT/-:HYPN/то:POST
```

## 3. Configuration

The experiments are controlled by two JSON configuration files: `models.json` and `datasets.json`.

### `models.json`

This file defines the BERT-like models you want to use. You can specify the model name from the Hugging Face Hub and its corresponding learning rate.

```json
{
  "bert-base": {
    "model_type": "bert",
    "model_name": "bert-base-cased",
    "learning_rate": 5e-5
  },
  "hplt": {
    "model_type": "bert",
    "model_name": "HPLT/hplt_bert_base_2_0_bel-Cyrl",
    "learning_rate": 5e-5
  }
}
```

### `datasets.json`

This file defines the datasets. The `corpus_dir` should match the folder name inside the `data/` directory.

```json
{
  "my_slavic_corpus": {
    "corpus_dir": "your_corpus_name",
    "language": "Russian"
  }
}
```

## 4. Running Experiments

You can use a runner script `run.py` to easily manage experiments. This script parse command-line arguments and call the `MorphBERT` class.

### Training Command

To train a model, run the script with the `train` action.

```bash
python run_experiment.py \
    --model_config hplt \
    --dataset_config my_slavic_corpus \
    --fold 0 \
    --action train \
    --epochs 30
```

-   This command will train the `hplt` model on `fold 0` of the `my_slavic_corpus` dataset for 30 epochs.
-   Trained model artifacts will be saved to `outputs/my_slavic_corpus/hplt/lemma/no_lemma/0/`.

### Prediction Command

To run inference using a fine-tuned model, use the `predict` action.

```bash
python run_experiment.py \
    --model_config hplt \
    --dataset_config my_slavic_corpus \
    --fold 0 \
    --action predict
```

-   This command will load the fine-tuned model from the output directory and run predictions on the test set for `fold 0`.
-   A CSV file with predictions (`0.csv`) and a JSON file with evaluation metrics (`metrics.json`) will be saved to `results/my_slavic_corpus/hplt/lemma/no_lemma/`.

## Citation

The code is based on the research presented in the paper [**BERT-like Models for Slavic Morpheme Segmentation**](https://aclanthology.org/2025.acl-long.337/). If you use this code in your research, please cite our paper.

```bibtex
@inproceedings{morozov-etal-2025-bert,
    title = "{BERT}-like Models for {S}lavic Morpheme Segmentation",
    author = "Morozov, Dmitry  and
      Astapenka, Lizaveta  and
      Glazkova, Anna  and
      Garipov, Timur  and
      Lyashevskaya, Olga",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.337/",
    doi = "10.18653/v1/2025.acl-long.337",
    pages = "6795--6815",
    ISBN = "979-8-89176-251-0"
}
```