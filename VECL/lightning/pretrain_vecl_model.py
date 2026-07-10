from .. import builder
from pytorch_lightning.core import LightningModule


class Pretrain_VECL_Model(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.VECL_model = builder.build_VECL_model(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.VECL_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        return loss

    def shared_step(self, batch, split):

        i2t_cls, t2i_cls = self.VECL_model(batch)
        loss = self.VECL_model.calc_loss(i2t_cls, t2i_cls, batch["label_sample"], batch["label_list"])

        # log training progress
        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )

        return loss