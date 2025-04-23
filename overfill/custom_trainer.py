from trl import SFTTrainer
import gc
import torch
import os

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, eval_data_collator=None, eval_refresh_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator or self.data_collator
        self.eval_refresh_interval = eval_refresh_interval
        
        # Set wandb project and entity if specified
        if hasattr(self.args, 'wandb_project') and self.args.wandb_project is not None:
            os.environ['WANDB_PROJECT'] = self.args.wandb_project
        if hasattr(self.args, 'wandb_entity') and self.args.wandb_entity is not None:
            os.environ['WANDB_ENTITY'] = self.args.wandb_entity

    def get_eval_dataloader(self, eval_dataset=None):
        # Save original data collator
        original_collator = self.data_collator
        # Use eval collator
        self.data_collator = self.eval_data_collator
        # Get eval dataloader
        dataloader = super().get_eval_dataloader(eval_dataset)
        # Restore original collator
        self.data_collator = original_collator
        return dataloader

    def prediction_step(self, model, inputs, prediction_loss_only, **kwargs):
        # Save original refresh interval
        original_refresh_interval = model.refresh_interval
        # Set evaluation refresh interval
        model.refresh_interval = self.eval_refresh_interval
        
        try:
            # Run prediction with refresh
            outputs = super().prediction_step(model, inputs, prediction_loss_only, **kwargs)
        finally:
            # Restore original refresh interval
            model.refresh_interval = original_refresh_interval
            
        return outputs
    
    def training_step(self, model, *args, **kwargs):
        loss = super().training_step(model, *args, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()
        # get_accelerator().empty_cache()
        return loss
