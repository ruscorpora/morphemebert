import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from pathlib import Path
import pickle
import pandas as pd
import json


from morph_base import MorphBase

# Внутренние классы для PyTorch (можно оставить их здесь)
class ResNetBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class CharResNetModel(nn.Module):
    def __init__(self, vocab_size, num_labels, hidden_dim=256, num_layers=10, kernel_size=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.dropout_emb = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ResNetBlock1D(hidden_dim, kernel_size, dropout) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = self.dropout_emb(x)
        x = x.permute(0, 2, 1) # (Batch, Hidden, Seq)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1) # (Batch, Seq, Hidden)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return logits, loss

class CharTokenizer:
    def __init__(self):
        self.char2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2char = {0: "<PAD>", 1: "<UNK>"}
    
    def build_vocab(self, texts):
        chars = set()
        for text in texts:
            chars.update(list(text))
        for char in sorted(list(chars)):
            if char not in self.char2id:
                idx = len(self.char2id)
                self.char2id[char] = idx
                self.id2char[idx] = char
                
    def __len__(self): return len(self.char2id)
    def encode(self, text): return [self.char2id.get(c, self.char2id["<UNK>"]) for c in text]
    def save(self, path): 
        with open(path, 'w', encoding='utf-8') as f: json.dump(self.char2id, f, ensure_ascii=False, indent=2)
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f: self.char2id = json.load(f)
        self.id2char = {v: k for k, v in self.char2id.items()}

class MorphDataset(Dataset):
    def __init__(self, token_ids, label_ids):
        self.token_ids = token_ids
        self.label_ids = label_ids
    def __len__(self): return len(self.token_ids)
    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.token_ids[idx], dtype=torch.long), "labels": torch.tensor(self.label_ids[idx], dtype=torch.long)}

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    return input_ids, labels

class MorphResNet(MorphBase):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.tokenizer = CharTokenizer()

    def _prepare_dataset(self, file_path: Path):
        token_ids_list, label_ids_list, lemmas_list = [], [], []
        
        # Сначала соберем вокабуляр, если это тренировка (или просто тексты)
        # Но для простоты в ResNet мы строим вокабуляр в train отдельно
        
        for lemma, parsing in self._read_pairs(file_path):
            lemma_processed = lemma.replace("ё", "е")
            tokens = list(lemma_processed)
            bmes = self._convert2bmes(parsing)
            
            if not bmes: continue
            if len(tokens) != len(bmes): continue

            input_ids = self.tokenizer.encode(lemma_processed)
            # При prediction label_to_id может не быть, если модель еще не загружена, 
            # но _prepare_dataset вызывается внутри train/predict после загрузки
            label_ids = [self.label_to_id.get(l, -100) for l in bmes]
            
            token_ids_list.append(input_ids)
            label_ids_list.append(label_ids)
            lemmas_list.append(lemma)
            
        return MorphDataset(token_ids_list, label_ids_list), lemmas_list

    def train(self, fold: int):
        train_file = self.data_dir / self.config["train_file"].format(fold=fold, mode=self.mode.value)
        outputs_dir = self._get_output_path(fold)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Train file: {train_file}")

        # 1. Labels & Vocab
        self._determine_labels_from_file(train_file)
        
        # Build Char Vocab specific logic
        all_lemmas = []
        for lemma, _ in self._read_pairs(train_file):
            all_lemmas.append(lemma.replace("ё", "е"))
        self.tokenizer.build_vocab(all_lemmas)
        print(f"Vocab size: {len(self.tokenizer)}")

        # 2. Dataset
        train_dataset, _ = self._prepare_dataset(train_file)
        train_loader = DataLoader(train_dataset, batch_size=self.config.get("batch_size", 32), shuffle=True, collate_fn=collate_fn)

        model = CharResNetModel(
            vocab_size=len(self.tokenizer), num_labels=len(self.label_list),
            hidden_dim=self.config.get("hidden_dim", 256), num_layers=self.config.get("num_layers", 10),
            dropout=self.config.get("dropout", 0.1)
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.get("learning_rate", 1e-3))
        
        model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_ids, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                batch_ids, batch_labels = batch_ids.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                _, loss = model(batch_ids, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

        torch.save(model.state_dict(), outputs_dir / "model.pt")
        self.tokenizer.save(outputs_dir / "vocab.json")
        with open(outputs_dir / "labels.json", "w") as f:
            json.dump({"label_list": self.label_list, "label_to_id": self.label_to_id}, f)
        
        config_to_save = {
            "vocab_size": len(self.tokenizer), "num_labels": len(self.label_list),
            "hidden_dim": self.config.get("hidden_dim", 256), "num_layers": self.config.get("num_layers", 10),
            "dropout": self.config.get("dropout", 0.1)
        }
        with open(outputs_dir / "config.json", "w") as f: json.dump(config_to_save, f)
        print("Training complete.")

    def predict(self, fold: int):
        models_dir = self._get_output_path(fold)
        test_file = self.data_dir / self.config["test_file"].format(fold=fold, mode=self.mode.value)
        results_dir = self._get_results_path()
        predicted_file = results_dir / f"{fold}.csv"

        print(f"Loading model from: {models_dir}")
        self.tokenizer.load(models_dir / "vocab.json")
        with open(models_dir / "labels.json", "r") as f:
            data = json.load(f)
            self.label_list, self.label_to_id = data["label_list"], data["label_to_id"]
        with open(models_dir / "config.json", "r") as f: model_params = json.load(f)

        model = CharResNetModel(**model_params).to(self.device)
        model.load_state_dict(torch.load(models_dir / "model.pt", map_location=self.device))
        model.eval()

        test_dataset, test_lemmas = self._prepare_dataset(test_file)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        # Чтобы получить target_text, перечитаем файл (так как prepare_dataset не возвращает сырой parsing)
        target_text = []
        for lemma, parsing in self._read_pairs(test_file):
            if not self._convert2bmes(parsing): continue # filter invalid
            if len(list(lemma.replace("ё", "е"))) != len(self._convert2bmes(parsing)): continue
            target_text.append(parsing)

        all_preds = []
        with torch.no_grad():
            for batch_ids, _ in tqdm(test_loader):
                batch_ids = batch_ids.to(self.device)
                logits, _ = model(batch_ids, None)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                for i in range(len(preds)):
                    # Реальная длина (без паддинга) определяется токеном != 0 или просто по длине леммы из списка
                    # Здесь просто берем длину входного токенизированного слова (без 0)
                    length = torch.sum(batch_ids[i] != 0).item()
                    pred_seq = [self.label_list[p] for p in preds[i][:length]]
                    all_preds.append(pred_seq)

        predicted_text = [self._convert2parsing(lemma, bmes) for lemma, bmes in zip(test_lemmas, all_preds)]

        test_df = pd.DataFrame({
            "input_text": test_lemmas,
            "target_text": target_text,
            "predicted_text": predicted_text
        })
        test_df.to_csv(predicted_file, index=False)
        
        metrics = self._measure_quality(target_text, predicted_text)
        print("Metrics:", metrics)
        self._save_metrics(metrics, fold)

