import argparse
from morphbert import MorphBERT, SplitMode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict with MorphBERT.")
    parser.add_argument("--model_config", type=str, required=True, help="Name of the model config from models.json")
    parser.add_argument("--dataset_config", type=str, required=True, help="Name of the dataset config from datasets.json")
    parser.add_argument("--fold", type=int, required=True, help="Fold number for cross-validation.")
    parser.add_argument("--action", type=str, choices=["train", "predict"], required=True, help="Action to perform.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--use_lemma", action="store_true", help="Whether to use the lemma as a special token.")
    parser.add_argument("--mode", type=str, default="lemma", choices=["form", "lemma", "roots"], help="Split mode for data.")

    args = parser.parse_args()

    morph_bert_model = MorphBERT(
        model_config_name=args.model_config,
        dataset_config_name=args.dataset_config,
        num_epochs=args.epochs,
        use_lemma=args.use_lemma,
        mode=SplitMode(args.mode)
    )

    if args.action == "train":
        print(f"--- Starting Training for Fold {args.fold} ---")
        morph_bert_model.train(fold=args.fold)
        print("--- Training Finished ---")
    elif args.action == "predict":
        print(f"--- Starting Prediction for Fold {args.fold} ---")
        morph_bert_model.predict(fold=args.fold)
        print("--- Prediction Finished ---")
