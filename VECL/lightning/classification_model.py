import torch
import os
from .. import builder
from pytorch_lightning.core import LightningModule


class ClassificationModel(LightningModule):
    """Pytorch-Lightning Module"""
    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()
        self.cfg_classify = cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if os.path.exists(self.cfg_classify.model.ckpt_path):
            ckpt = torch.load(self.cfg_classify.model.ckpt_path, map_location=device)
            self.cfg_ckpt = ckpt["hyper_parameters"]
            del ckpt
        else:
            self.cfg_ckpt = None
        self.classification_model = builder.build_classification_model(self.cfg_ckpt, self.cfg_classify, cfg)
        self.lr = self.cfg_classify.lightning.trainer.lr
        self.dm = None

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg_classify, self.lr, self.classification_model)
        scheduler = builder.build_scheduler(self.cfg_classify, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def shared_step(self, batch, split):
        """Similar to traning step"""

        image, texts, label, caption_ids, attention_mask, token_type_ids = batch["images"], batch["texts"], batch["labels"], batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"]

        i2t_cls, t2i_cls = self.classification_model(image, texts, caption_ids, attention_mask, token_type_ids)

        loss = self.classification_model.calc_loss(i2t_cls, t2i_cls, label)

        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )

        return_dict = {"loss": loss, "logit": i2t_cls + t2i_cls, "label": label}
        return return_dict