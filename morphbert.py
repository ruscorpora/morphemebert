import logging
import json
from pathlib import Path
import sys
from enum import Enum
from collections import defaultdict

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


class SplitMode(Enum):
    SPLIT_BY_FORM = "form"
    SPLIT_BY_LEMMA = "lemma"
    SPLIT_BY_ROOTS = "roots"


class MorphBERT:
    """
    A class to handle fine-tuning and inference of BERT-like models for morpheme segmentation
    using separate model and dataset configurations.
    """

    def __init__(
        self,
        model_config_name: str,
        dataset_config_name: str,
        models_config_path: str = "models.json",
        datasets_config_path: str = "datasets.json",
        num_epochs: int = 30,
        outputs_dir: str = "outputs",
        data_dir: str = "data",
        results_dir: str = "results",
        use_lemma: bool = False,
        mode: SplitMode = SplitMode.SPLIT_BY_LEMMA,
    ):
        """
        Initializes the MorphBERT model with given configurations.

        Args:
            model_config_name (str): The name of the model config to use (from models.json).
            dataset_config_name (str): The name of the dataset config to use (from datasets.json).
            models_config_path (str): Path to the JSON file with model configurations.
            datasets_config_path (str): Path to the JSON file with dataset configurations.
            num_epochs (int): Number of training epochs.
            outputs_dir (str): The base directory to save model outputs.
            data_dir (str): The base directory where data is stored.
        """
        self.model_config_name = model_config_name
        self.dataset_config_name = dataset_config_name
        self.num_epochs = num_epochs
        self.outputs_dir = Path(outputs_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)

        try:
            with open(models_config_path, "r") as f:
                model_conf = json.load(f)[model_config_name]
        except KeyError:
            print(
                f"Error: Model configuration '{model_config_name}' not found in '{models_config_path}'"
            )
            sys.exit(1)

        try:
            with open(datasets_config_path, "r") as f:
                dataset_conf = json.load(f)[dataset_config_name]
        except KeyError:
            print(
                f"Error: Dataset configuration '{dataset_config_name}' not found in '{datasets_config_path}'"
            )
            sys.exit(1)

        self.config = {**model_conf, **dataset_conf}

        self.use_lemma = use_lemma
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )
        self.label_list = self._get_label_list()
        self.label_to_id = {l: i for i, l in enumerate(self.label_list)}

    def _get_label_list(self):
        custom_labels = []
        if self.use_lemma:
            custom_labels.append("0")
        for mtype in ["ROOT", "PREF", "SUFF", "END", "POST", "LINK", "HYPN"]:
            custom_labels.extend(
                [f"B-{mtype}", f"M-{mtype}", f"E-{mtype}", f"S-{mtype}"]
            )
        return custom_labels

    @staticmethod
    def _convert2bmes(parsing: str) -> list:
        if parsing in ["FAILED", "WRONG_LEN"]:
            return []
        bmes = []
        for m in parsing.split("/"):
            m_str, m_type = m.split(":")
            if len(m_str) == 1:
                bmes.append(f"S-{m_type}")
            else:
                bmes.append(f"B-{m_type}")
                bmes.extend([f"M-{m_type}"] * (len(m_str) - 2))
                bmes.append(f"E-{m_type}")
        return bmes

    @staticmethod
    def _convert2parsing(lemma: str, bmes: list) -> str:
        parsing = []
        current_mtext = ""
        current_mtype = ""
        for letter, label in zip(lemma, bmes):
            if not "-" in label:
                parsing.append(f"{letter}:UNK")
                current_mtext, current_mtype = "", ""
                continue
            pos, mtype = label.split("-")
            if pos == "S":
                if current_mtext:
                    parsing.append(f"{current_mtext}:{current_mtype}")
                parsing.append(f"{letter}:{mtype}")
                current_mtext, current_mtype = "", ""
            elif pos == "B":
                if current_mtext:
                    parsing.append(f"{current_mtext}:{current_mtype}")
                current_mtext = letter
                current_mtype = mtype
            else:
                current_mtext += letter
        if current_mtext:
            parsing.append(f"{current_mtext}:{current_mtype}")
        return "/".join(parsing)

    def _load_data_for_hf(self, file_path: Path) -> Dataset:
        pairs = [x.strip().split("\t")[:2] for x in open(file_path, "r").readlines()]
        data = {"tokens": [], "labels": [], "lemmas": []}
        for lemma, parsing in pairs:
            lemma_processed = lemma.replace("ё", "е")
            tokens = list(lemma_processed)
            bmes = self._convert2bmes(parsing)

            if self.use_lemma:
                tokens = [lemma_processed] + tokens
                bmes = ["0"] + bmes

            data["tokens"].append(tokens)
            data["labels"].append(bmes)
            data["lemmas"].append(lemma)
        return Dataset.from_dict(data)

    def _tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def _lemma_suffix(self):
        return "with_lemma" if self.use_lemma else "no_lemma"

    def _get_output_path(self, fold: int) -> Path:
        """Generates a descriptive path for model outputs."""
        return (
            self.outputs_dir
            / self.dataset_config_name
            / self.model_config_name
            / self.mode.value
            / self._lemma_suffix()
            / str(fold)
        )

    @staticmethod
    def _f_measure(targets, predicted, SE):
        TP, FP, FN, equal, total = 0, 0, 0, 0, 0
        corr_words = 0
        for corr, pred in zip(targets, predicted):
            corr_len = len(corr)
            pred_len = len(pred)
            boundaries = [i for i in range(corr_len) if corr[i] in SE]
            pred_boundaries = [i for i in range(pred_len) if pred[i] in SE]
            common = [x for x in boundaries if x in pred_boundaries]
            TP += len(common)
            FN += len(boundaries) - len(common)
            FP += len(pred_boundaries) - len(common)
            equal += sum(int(x == y) for x, y in zip(corr, pred))
            total += len(corr)
            corr_words += corr == pred
        return TP, FP, FN, equal, total, corr_words

    def _measure_quality(self, targets, predicted):
        targets = [self._convert2bmes(x) for x in targets]
        predicted = [self._convert2bmes(x) for x in predicted]

        SE = [
            "{}-{}".format(x, y)
            for x in "SE"
            for y in ["ROOT", "PREF", "SUFF", "END", "LINK"]
        ]
        TP, FP, FN, equal, total, corr_words = self._f_measure(targets, predicted, SE)

        results = {
            "Precision_full": TP / (TP + FP) if (TP + FP) > 0 else 0,
            "Recall_full": TP / (TP + FN) if (TP + FN) > 0 else 0,
            "F1_full": TP / (TP + 0.5 * (FP + FN)) if (TP + 0.5 * (FP + FN)) > 0 else 0,
            "Accuracy": equal / total if total > 0 else 0,
            "WordAccuracy": corr_words / len(targets) if len(targets) > 0 else 0,
        }

        SE = ["{}-{}".format(x, y) for x in "SE" for y in ["ROOT"]]
        TP, FP, FN, equal, total, corr_words = self._f_measure(targets, predicted, SE)
        results["Precision_root"] = TP / (TP + FP) if (TP + FP) > 0 else 0
        results["Recall_root"] = TP / (TP + FN) if (TP + FN) > 0 else 0
        results["F1_root"] = (
            TP / (TP + 0.5 * (FP + FN)) if (TP + 0.5 * (FP + FN)) > 0 else 0
        )

        return results

    def train(self, fold: int):
        """
        Fine-tunes the BERT-like model.

        Args:
            fold (int): The fold number for training.
        """
        train_file = (
            self.data_dir
            / f"{self.config['corpus_dir']}/split-by-{self.mode.value}/{str(fold)}/train.txt"
        )
        outputs_dir = self._get_output_path(fold)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        print(f"Train file: {train_file}")
        print(f"Outputs dir: {outputs_dir}")

        train_dataset = self._load_data_for_hf(train_file)
        tokenized_train_dataset = train_dataset.map(
            self._tokenize_and_align_labels, batched=True
        )

        model = AutoModelForTokenClassification.from_pretrained(
            self.config["model_name"],
            num_labels=len(self.label_list),
            id2label={i: l for i, l in enumerate(self.label_list)},
            label2id=self.label_to_id,
            trust_remote_code=(self.model_config_name == "hplt"),
            use_safetensors=(self.model_config_name not in ["hplt", "bel_roberta"]),
        )

        training_args = TrainingArguments(
            output_dir=str(outputs_dir),
            num_train_epochs=self.num_epochs,
            learning_rate=self.config["learning_rate"],
            per_device_train_batch_size=16,
            overwrite_output_dir=True,
            save_steps=-1,
            save_total_limit=1,
            logging_steps=100,
            fp16=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        trainer.train()
        trainer.save_model()
        print("Training complete.")

    def predict(self, fold: int):
        """
        Infers the fine-tuned model on a test set.

        Args:
            fold (int): The fold number for prediction.
        """
        models_dir = self._get_output_path(fold)
        test_file = (
            self.data_dir
            / f"{self.config['corpus_dir']}/split-by-{self.mode.value}/{str(fold)}/test.txt"
        )

        results_dir = (
            self.results_dir
            / self.dataset_config_name
            / self.model_config_name
            / self.mode.value
            / self._lemma_suffix()
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        predicted_file = results_dir / f"{fold}.csv"

        print(f"Loading model from: {models_dir}")
        print(f"Test file: {test_file}")
        print(f"Predictions will be saved to: {predicted_file}")

        test_dataset = self._load_data_for_hf(test_file)
        tokenized_test_dataset = test_dataset.map(
            self._tokenize_and_align_labels, batched=True
        )

        model = AutoModelForTokenClassification.from_pretrained(
            str(models_dir),
            trust_remote_code=(self.model_config_name == "hplt"),
        )

        trainer = Trainer(
            model=model,
            data_collator=self.data_collator,
        )

        predictions, _, _ = trainer.predict(tokenized_test_dataset)
        predicted_labels = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predicted_labels, tokenized_test_dataset["labels"])
        ]

        predicted_text = [
            self._convert2parsing(lemma, bmes[1:] if self.use_lemma else bmes)
            for lemma, bmes in zip(test_dataset["lemmas"], true_predictions)
        ]

        test_df = pd.DataFrame({
            "input_text": test_dataset["lemmas"],
            "target_text": [
                self._convert2parsing(p['tokens'][1:], p['labels'][1:])
                if self.use_lemma else
                self._convert2parsing(p['tokens'], p['labels'])
                for p in test_dataset
            ]
        })
        test_df["predicted_text"] = predicted_text
        test_df.to_csv(predicted_file, index=False)

        print("Prediction complete.")

        metrics = self._measure_quality(test_df.target_text.to_list(), predicted_text)
        print("Metrics:")
        print(metrics)

        metrics_file = results_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
        else:
            metrics_data = defaultdict(list)
        for k, v in metrics.items():
            metrics_data[k].append(v)
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=1)

