from src.contrastive_learning import contrastive_pretrain_pipeline
from src.supervised_fine_tune import supervised_fine_tune_pipeline


def run_pipeline():
    # Run contrastive pre-training
    print("Starting contrastive pre-training...")
    contrastive_pretrain_pipeline()

    # Run supervised fine-tuning
    print("Starting supervised fine-tuning...")
    supervised_fine_tune_pipeline()

    print("Training pipeline completed. Models weights saved.")


if __name__ == "__main__":
    run_pipeline()
