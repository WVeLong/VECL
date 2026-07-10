from .pretrain_vecl_model import Pretrain_VECL_Model
from .classification_model import ClassificationModel


LIGHTNING_MODULES = {
    "pretrain": Pretrain_VECL_Model,
    "finetune_classification": ClassificationModel
}