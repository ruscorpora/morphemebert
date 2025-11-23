import logging
import json
import pickle
import sys
from pathlib import Path
from enum import Enum
from collections import defaultdict
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SplitMode(Enum):
    SPLIT_BY_FORM = "form"
    SPLIT_BY_LEMMA = "lemma"
    SPLIT_BY_ROOTS = "roots"
    SPLIT_BY_RANDOM = "random"

class MorphBase:
    """
    Base class for Morphological Segmentation models.
    Handles configuration, data utilities, and metrics.
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
        mode: SplitMode = SplitMode.SPLIT_BY_RANDOM,
    ):
        self.model_config_name = model_config_name
        self.dataset_config_name = dataset_config_name
        self.num_epochs = num_epochs
        self.outputs_dir = Path(outputs_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.use_lemma = use_lemma
        self.mode = mode

        # Load Configurations
        try:
            with open(models_config_path, "r") as f:
                model_conf = json.load(f)[model_config_name]
        except KeyError:
            print(f"Error: Model configuration '{model_config_name}' not found in '{models_config_path}'")
            sys.exit(1)

        try:
            with open(datasets_config_path, "r") as f:
                dataset_conf = json.load(f)[dataset_config_name]
        except KeyError:
            print(f"Error: Dataset configuration '{dataset_config_name}' not found in '{datasets_config_path}'")
            sys.exit(1)

        self.config = {**model_conf, **dataset_conf}
        
        self.label_list = None
        self.label_to_id = None

    # --- File & Path Utilities ---

    def _lemma_suffix(self):
        return "with_lemma" if self.use_lemma else "no_lemma"

    def _get_output_path(self, fold: int) -> Path:
        return (
            self.outputs_dir
            / self.dataset_config_name
            / self.model_config_name
            / self.mode.value
            / self._lemma_suffix()
            / str(fold)
        )
    
    def _get_results_path(self) -> Path:
        path = (
            self.results_dir
            / self.dataset_config_name
            / self.model_config_name
            / self.mode.value
            / self._lemma_suffix()
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _read_pairs(self, file_path: Path):
        """Generator that yields (lemma, parsing) tuples from a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                yield parts[0], parts[1]

    # --- Label & Parsing Utilities ---

    @staticmethod
    def _convert2bmes(parsing: str) -> list:
        if parsing in ["FAILED", "WRONG_LEN"]:
            return []
        bmes = []
        for m in parsing.split("/"):
            if ":" not in m:
                continue
            m_str, m_type = m.rsplit(":", 1)
            
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

    def _determine_labels_from_file(self, file_path: Path):
        """Scans a file to build the label_list based on unique morpheme types."""
        unique_types = set()
        for _, parsing in self._read_pairs(file_path):
            if parsing in ["FAILED", "WRONG_LEN"]:
                continue
            try:
                for m in parsing.split("/"):
                    if ":" in m:
                        _, m_type = m.rsplit(":", 1)
                        unique_types.add(m_type)
            except ValueError:
                continue
        
        sorted_types = sorted(list(unique_types))
        custom_labels = []
        if self.use_lemma:
            custom_labels.append("0") # Special token for BERT-like if using lemma
        
        for mtype in sorted_types:
            custom_labels.extend(
                [f"B-{mtype}", f"M-{mtype}", f"E-{mtype}", f"S-{mtype}"]
            )
        
        self.label_list = custom_labels
        self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
        print(f"Detected {len(sorted_types)} unique morpheme types. Total labels: {len(self.label_list)}")

    # --- Metrics (From MorphBERT) ---

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

        # Используем список из MorphBERT
        SE = [
            "{}-{}".format(x, y)
            for x in "SE"
            for y in ["ROOT", "PREF", "SUFF", "END", "LINK", "STEM", "POST", "X"] 
        ]
        TP, FP, FN, equal, total, corr_words = self._f_measure(targets, predicted, SE)

        results = {
            "Precision": TP / (TP + FP) if (TP + FP) > 0 else 0,
            "Recall": TP / (TP + FN) if (TP + FN) > 0 else 0,
            "F1": TP / (TP + 0.5 * (FP + FN)) if (TP + 0.5 * (FP + FN)) > 0 else 0,
            "Accuracy": equal / total if total > 0 else 0,
            "WordAccuracy": corr_words / len(targets) if len(targets) > 0 else 0,
        }
        return results

    def _save_metrics(self, metrics, fold):
        results_dir = self._get_results_path()
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

    # --- Abstract Methods ---
    def train(self, fold: int):
        raise NotImplementedError

    def predict(self, fold: int):
        raise NotImplementedError

