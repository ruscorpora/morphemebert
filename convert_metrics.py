import json
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert metrics values to table format.")
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the model configuration from models.json')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset configuration from datasets.json')
    parser.add_argument('--mode', type=str, default='lemma',
                        help='One of split modes: by form, by lemma, by roots')
    parser.add_argument('--use_lemma', action='store_true', default=False,
                        help='Add lemma to the features')
    args = parser.parse_args()

    lemma_suffix = "with_lemma" if args.use_lemma else "no_lemma"
    metrics_file = Path('results') / args.dataset / args.model / args.mode / lemma_suffix / "metrics.json"
    metrics_data = json.load(metrics_file.open("r"))

#    for idx in range(len(metrics_data["WordAccuracy"])):
#        for metric_label in ["Precision", "Recall", "F1", "Accuracy", "WordAccuracy"]:
#            print(f"{metrics_data[metric_label][idx] * 100.:.2f}")
#        print("---")
    for metric_label in ["Precision", "Recall", "F1", "Accuracy", "WordAccuracy"]:
        print(",".join(f"{metrics_data[metric_label][idx] * 100.:.2f}" for idx in range(len(metrics_data["WordAccuracy"]))))

