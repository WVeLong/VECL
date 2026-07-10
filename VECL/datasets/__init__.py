from . import data_module
from . import pretraining_dataset


DATA_MODULES = {
    "pretrain": data_module.PretrainingDataModule,
    "finetune": data_module.FinetuneDataModule,
}
