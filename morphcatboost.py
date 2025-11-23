from catboost import CatBoostClassifier, Pool

from pathlib import Path
import pandas as pd
import json
import pickle

from morph_base import MorphBase

class MorphCatBoost(MorphBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.window_size = self.config.get("window_size", 3)
        self.id_to_label = None

    def _extract_features_for_word(self, word):
        word_len = len(word)
        features_list = []
        for idx, char in enumerate(word):
            feat = {}
            feat['c_0'] = char
            for w in range(1, self.window_size + 1):
                feat[f'c_L{w}'] = word[idx - w] if idx - w >= 0 else "<PAD>"
                feat[f'c_R{w}'] = word[idx + w] if idx + w < word_len else "<PAD>"
            feat['pos_rel'] = idx / max(word_len, 1)
            feat['len'] = word_len
            features_list.append(feat)
        return features_list

    def _prepare_tabular_dataset(self, file_path: Path, is_train: bool = True):
        all_features, all_labels, lemmas = [], [], []
        
        for lemma, parsing in self._read_pairs(file_path):
            lemma_processed = lemma.replace("ё", "е")
            tokens = list(lemma_processed)
            bmes = self._convert2bmes(parsing)
            
            if not bmes: continue
            if len(tokens) != len(bmes): continue
                
            lemmas.append(lemma)
            word_feats = self._extract_features_for_word(tokens)
            all_features.extend(word_feats)
            
            if is_train:
                label_ids = [self.label_to_id.get(l, -1) for l in bmes]
                all_labels.extend(label_ids)
        
        df = pd.DataFrame(all_features)
        return (df, all_labels, lemmas) if is_train else (df, lemmas)

    def train(self, fold: int):
        train_file = self.data_dir / self.config["train_file"].format(fold=fold, mode=self.mode.value)
        outputs_dir = self._get_output_path(fold)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Train file: {train_file}")

        self._determine_labels_from_file(train_file)
        self.id_to_label = {i: l for l, i in self.label_to_id.items()}

        print("Preparing tabular data...")
        X_train, y_train, _ = self._prepare_tabular_dataset(train_file, is_train=True)
        cat_features = [col for col in X_train.columns if col.startswith('c_')]
        
        model = CatBoostClassifier(
            iterations=self.config.get("iterations", self.num_epochs),
            learning_rate=self.config.get("learning_rate", 0.1),
            depth=self.config.get("depth", 6),
            loss_function='MultiClass',
            verbose=100,
            task_type=self.config.get("task_type", "CPU"),
            early_stopping_rounds=50
        )
        
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        model.fit(train_pool)
        
        model.save_model(str(outputs_dir / "model.cbm"))
        with open(outputs_dir / "labels.pkl", "wb") as f:
            pickle.dump({"label_list": self.label_list, "label_to_id": self.label_to_id, "id_to_label": self.id_to_label}, f)
        print("Training complete.")

    def predict(self, fold: int):
        models_dir = self._get_output_path(fold)
        test_file = self.data_dir / self.config["test_file"].format(fold=fold, mode=self.mode.value)
        results_dir = self._get_results_path()
        predicted_file = results_dir / f"{fold}.csv"

        print(f"Loading model from: {models_dir}")
        with open(models_dir / "labels.pkl", "rb") as f:
            meta = pickle.load(f)
            self.label_list, self.label_to_id, self.id_to_label = meta["label_list"], meta["label_to_id"], meta["id_to_label"]
            
        model = CatBoostClassifier()
        model.load_model(str(models_dir / "model.cbm"))

        print("Preparing test data...")
        df_test, test_lemmas = self._prepare_tabular_dataset(test_file, is_train=False)
        
        # Получаем word_lengths для реконструкции
        # Важно: _prepare_tabular_dataset фильтрует данные, поэтому мы должны пересчитать длины
        # по тем леммам, которые попали в выборку (test_lemmas)
        word_lengths = [len(l.replace("ё", "е")) for l in test_lemmas]

        print(f"Predicting on {len(df_test)} chars...")
        preds = model.predict(df_test).flatten()
        
        predicted_texts = []
        current_idx = 0
        for length in word_lengths:
            word_pred_ids = preds[current_idx : current_idx + length]
            current_idx += length
            bmes_pred = [self.id_to_label[int(pid)] for pid in word_pred_ids]
            lemma_str = test_lemmas[len(predicted_texts)].replace("ё", "е")
            predicted_texts.append(self._convert2parsing(lemma_str, bmes_pred))
            
        # Target texts for metrics (нужно снова прочитать, так как prepare их не возвращает)
        target_text = []
        for lemma, parsing in self._read_pairs(test_file):
             if not self._convert2bmes(parsing): continue
             if len(list(lemma.replace("ё", "е"))) != len(self._convert2bmes(parsing)): continue
             target_text.append(parsing)

        test_df = pd.DataFrame({
            "input_text": test_lemmas,
            "target_text": target_text,
            "predicted_text": predicted_texts
        })
        test_df.to_csv(predicted_file, index=False)
        
        metrics = self._measure_quality(target_text, predicted_texts)
        print("Metrics:", metrics)
        self._save_metrics(metrics, fold)

