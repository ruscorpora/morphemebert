import argparse
import time
from morphbert import MorphBERT, SplitMode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run N-fold cross-validation for MorphBERT.")
    parser.add_argument('--model', type=str, required=True, help='Name of the model configuration from models.json')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset configuration from datasets.json')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs for each fold')
    parser.add_argument('--mode', type=str, default='lemma', help='One of split modes: by form, by lemma, by roots')
    parser.add_argument('--use_lemma', action='store_true', default=False, help='Add lemma to the features')
    parser.add_argument('--predict_only', action='store_true', default=False, help='Run cross-val without training')
    parser.add_argument('--only_fold', type=int, default=-1, help='If set, only runs fold with specific number')
    args = parser.parse_args()

    start_time = time.time()
    print(f"Starting {args.num_folds}-fold cross-validation for model '{args.model}' on dataset '{args.dataset}' in split-by-{args.mode} scenario, lemma used: {str(args.use_lemma)}.")

    for fold in range(1, args.num_folds + 1):
        if args.only_fold != -1 and fold != args.only_fold:
            continue
        fold_start_time = time.time()
        print("\n" + "=" * 50)
        print(f" FOLD {fold} of {args.num_folds}")
        print("=" * 50)

        # Initialize the model for the current fold
        morph_bert_model = MorphBERT(
            model_config_name=args.model,
            dataset_config_name=args.dataset,
            num_epochs=args.epochs,
            mode=SplitMode(args.mode),
            use_lemma=args.use_lemma
        )

        if not args.predict_only:
            # Train the model
            print(f"\n--- Training Fold {fold} ---")
            morph_bert_model.train(fold=fold)

        # Run prediction
        print(f"\n--- Predicting Fold {fold} ---")
        morph_bert_model.predict(fold=fold)

        fold_end_time = time.time()
        print(f"\nFold {fold} completed in {time.strftime('%H:%M:%S', time.gmtime(fold_end_time - fold_start_time))}.")

    end_time = time.time()
    print("\n" + "=" * 50)
    print("Cross-validation complete.")
    print(f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    print("=" * 50)
