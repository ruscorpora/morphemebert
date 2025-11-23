from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from pathlib import Path
import json
import pandas as pd
import numpy as np

from morph_base import MorphBase


class MorphBERT(MorphBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def _load_data_for_hf(self, file_path: Path) -> Dataset:
        data = {"tokens": [], "labels": [], "lemmas": []}
        
        for lemma, parsing in self._read_pairs(file_path):
            lemma_processed = lemma.replace("ё", "е")
            tokens = list(lemma_processed)
            bmes = self._convert2bmes(parsing)

            if not bmes: continue
            if len(tokens) != len(bmes):
                continue

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
                    label_ids.append(self.label_to_id.get(label[word_idx], -100))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def train(self, fold: int):
        train_file = self.data_dir / self.config["train_file"].format(fold=fold, mode=self.mode.value)
        outputs_dir = self._get_output_path(fold)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Train file: {train_file}")

        self._determine_labels_from_file(train_file)

        train_dataset = self._load_data_for_hf(train_file)
        tokenized_train_dataset = train_dataset.map(self._tokenize_and_align_labels, batched=True)

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
            logging_steps=1000,
            fp16=True,
            report_to="none",
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
        models_dir = self._get_output_path(fold)
        test_file = self.data_dir / self.config["test_file"].format(fold=fold, mode=self.mode.value)
        results_dir = self._get_results_path()
        predicted_file = results_dir / f"{fold}.csv"

        print(f"Loading model from: {models_dir}")
        model = AutoModelForTokenClassification.from_pretrained(
            str(models_dir),
            trust_remote_code=(self.model_config_name == "hplt"),
        )
        
        self.label_to_id = model.config.label2id
        self.label_list = [model.config.id2label[i] for i in sorted(model.config.id2label.keys())]

        test_dataset = self._load_data_for_hf(test_file)
        tokenized_test_dataset = test_dataset.map(self._tokenize_and_align_labels, batched=True)

        trainer = Trainer(model=model, data_collator=self.data_collator)
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
                self._convert2parsing(p['tokens'][1:], p['labels'][1:]) if self.use_lemma else self._convert2parsing(p['tokens'], p['labels'])
                for p in test_dataset
            ],
            "predicted_text": predicted_text
        })
        test_df.to_csv(predicted_file, index=False)
        
        metrics = self._measure_quality(test_df.target_text.to_list(), predicted_text)
        print("Metrics:", metrics)
        self._save_metrics(metrics, fold)

