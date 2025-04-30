BASE_MODEL_DIR = './models/pythia-70m-arxiv-scratch'
FINETUNED_MODEL_DIR = './models/pythia-70m-arxiv-finetuned'

# Original dataset
DATA_PATH = './datasets/pretraining/arxiv-metadata-oai-snapshot.jsonl'
# Tokenized but not grouped
TOKENIZED_DATA_PATH = './datasets/pretraining/tokenized_datasets'
# Finished processed dataset, ready for training
PROCESSED_DATA_PATH = './datasets/pretraining/grouped_datasets'

FINETUNED_DATA_PATH = './datasets/finetuning'