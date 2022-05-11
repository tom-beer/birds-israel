import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from lit_mlp import LitMLP
from data_module import birds, samples, batch_size
from loggings import ImagePredictionLogger

wandb_logger = WandbLogger(project="lit-wandb")

trainer = pl.Trainer(
    logger=wandb_logger,    # W&B integration
    log_every_n_steps=50,   # set the logging frequency
    gpus=0,                 # use all GPUs
    max_epochs=100,           # number of epochs
    deterministic=True,     # keep it deterministic
    callbacks=[ImagePredictionLogger(samples)]  # see Callbacks section
    )

# setup model
model = LitMLP(n_classes=18, batch_size=batch_size)

# fit the model
trainer.fit(model, birds)

# evaluate the model on a test set
trainer.test(datamodule=birds,
             ckpt_path=None)  # uses last-saved model

wandb.finish()
