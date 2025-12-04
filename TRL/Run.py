import argparse
import wandb
from selection.data_selection import SelectionConfig, select_training_data
from Train import TrainConfig, train_model

def run_pipeline(selection_config_file, train_config_file):
    wandb.login(key="0944191bcf43ea6231189f995e76d66cc523c13d")
    
    print("Start Data Selection")
    selection_config = SelectionConfig(selection_config_file)
    select_training_data(selection_config)
    
    print("Start Training Model")
    print("=" * 50)
    wandb.init(project="S2L_Cirriculum")
    train_config = TrainConfig(train_config_file)
    train_model(train_config)
    
    print("Pipeline done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection_config', type=str, required=True)
    parser.add_argument('--train_config', type=str, required=True)
    args = parser.parse_args()
    
    run_pipeline(args.selection_config, args.train_config)
