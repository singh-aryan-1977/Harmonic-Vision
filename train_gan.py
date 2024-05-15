import itertools

from src.pipeline import pipeline
from src.training_utils import training_utils

EXP_HPARAMS = {
    "params": ({},),
    "seeds": (420,),
}

DATASET = "custom"
DATA_PATH = "../input/custom-dataset"

def run_experiments():
    for hparams_overwrite_list, seed in itertools.product(EXP_HPARAMS["params"], EXP_HPARAMS["seeds"]):
        config = training_utils.get_config(DATASET)
        hparams_str = ""
        for k, v in hparams_overwrite_list.items():
            config[k] = v
            hparams_str += str(k) + "-" + str(v) + "_"
        config["model_architecture"] = "biggan" 
        config["hparams_str"] = hparams_str.strip("_")
        config["seed"] = seed
        run_experiment(config)


def run_experiment(config):
    training_utils.set_random_seed(seed=config.seed, device=config.device)
    training_pipeline = pipeline.GANPipeline.from_config(data_path=DATA_PATH, config=config)
    training_pipeline.train_model()


run_experiments()
