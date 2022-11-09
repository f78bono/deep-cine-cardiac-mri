import sys
sys.path.append('/path/to/source')

import os
import pathlib
import time
from argparse import ArgumentParser

import numpy as np
import torch
import pytorch_lightning as pl

from reconstruction.data import SliceDataset
from reconstruction.data.mri_data import fetch_dir
from reconstruction.data.subsample import create_mask_for_mask_type
from reconstruction.data.transforms import VarNetDataTransform
from reconstruction.pl_modules import MriDataModule, VarNetModule
from traintest_scripts.run_inference import InferenceTransform



def train_test_main(args, save_path):
    pl.seed_everything(args.seed)

    # ------------
    # DATA SECTION
    # ------------
    
    # This creates a k-space mask to subsample input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    test_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    
    # Data module - this handles data loaders
    data_module = MriDataModule(
        data_path=args.data_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=args.combine_train_val,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        use_dataset_cache_file=args.use_dataset_cache_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # -------------
    # MODEL SECTION
    # -------------
    
    # Load model state dictionary (generally for testing)
    if args.load_model:
        checkpoint_dir = args.default_root_dir / "checkpoints"
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            last_ckpt_path = str(ckpt_list[-1])
            print(f"Loading model from {last_ckpt_path}")    
            model = VarNetModule.load_from_checkpoint(last_ckpt_path)
        else:
            raise ValueError("No checkpoint available")
            
    else:
        # Build model
        model = VarNetModule(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            dynamic_type=args.dynamic_type,
            weight_sharing=args.weight_sharing,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
        )

    # ------------------
    # TRAIN-TEST SECTION
    # ------------------
    
    trainer = pl.Trainer.from_argparse_args(args)

    if args.mode == "train":
        
        print("Training VarNet "
                    f"{args.dynamic_type} with "
                    f"{args.num_cascades} cascades for "
                    f"{args.max_epochs} epochs.\nData is subsampled with a "
                    f"{args.mask_type} mask, acceleration "
                    f"{args.accelerations[0]}."
        )
        
        start_time = time.perf_counter()
        trainer.fit(model, datamodule=data_module)
        end_time = time.perf_counter()
        
        print(f"Training time: {(end_time-start_time) / 3600.} hours")
        
        if args.save_checkpoint:
            trainer.save_checkpoint(args.default_root_dir / f"checkpoints/varnet.ckpt")
            print(f"Saving checkpoint in varnet_{args.dynamic_type}_acc{args.accelerations[0]}_ckpt")
        
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")
        
    # -----------------
    # INFERENCE SECTION
    # -----------------
    
    if (args.mode == "test" and args.inference):
        
        inference_dataset = SliceDataset(
            root=args.data_path / "inference", transform=test_transform,
        )
        dataloader = torch.utils.data.DataLoader(inference_dataset, num_workers=2)
        inf_transform = InferenceTransform(model, 'varnet', save_path)
        time_for_inference = 0
        
        print('Starting inference..............')
        
        for batch in dataloader:
            with torch.no_grad():
                masked_kspace, mask, target, fname, _, _, _ = batch
                time_for_inference += inf_transform(masked_kspace, mask, target, fname)
                
        print(f"Elapsed time: {time_for_inference} seconds.")
    



def build_args():
    parser = ArgumentParser()

    # ----------
    # BASIC ARGS
    # ----------
    path_config = pathlib.Path("/root/traintest_scripts/dirs_path.yaml")
    backend = "dp"
    num_gpus = 2 if backend == "ddp" else 1
    batch_size = 1

    # Set defaults based on optional directory config
    data_path = fetch_dir("data_path", path_config)
    save_path = fetch_dir("save_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "varnet/varnet_logs"

    # -----------
    # CLIENT ARGS
    # -----------
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )
    
    parser.add_argument(
        "--epochs",
        default=150,
        type=int,
        help="Total number of epochs to train the model for",
    )
    
    parser.add_argument(
        "--save_checkpoint",
        default=0,
        choices=(0, 1),
        type=int,
        help="Whether to save a checkpoint of the model at the end of training",
    )
    
    parser.add_argument(
        "--resume_training",
        default=0,
        choices=(0, 1),
        type=int,
        help="Whether to resume training from the latest checkpoint",
    )
    
    parser.add_argument(
        "--load_model",
        default=0,
        choices=(0, 1),
        type=int,
        help="Whether to load the latest model in checkpoint dir, to be used for testing",
    )
    
    parser.add_argument(
        "--inference",
        default=1,
        choices=(0, 1),
        type=int,
        help="Whether to generate and save the reconstruction made by the trained model on an inference dataset",
    )

    # Data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[10],
        type=float,
        help="Number of central lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )
    

    # --------------
    # MODULES CONFIG
    # --------------
    
    # Data config
    parser = MriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,
        test_path=None,
        test_split="test",
        sample_rate=None,
        use_dataset_cache_file=True,
        combine_train_val=False,
        batch_size=batch_size,
        num_workers=4,
    )

    # Model config
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=10,
        pools=3,
        chans=16,
        sens_pools=3,
        sens_chans=8, 
        dynamic_type='XF',
        weight_sharing=False,
        lr=0.0001,  
        lr_step_size=140,
        lr_gamma=0.01,
        weight_decay=0.0,
    )
    
    args = parser.parse_args()
    
    # Configure checkpointing in checkpoint_dir
    checkpoint_dir = default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=default_root_dir / "checkpoints",
        filename=f"varnet_{args.dynamic_type}_acc{args.accelerations[0]}_ckpt",
        verbose=True,
        monitor="validation_loss",
        mode="min",
    )

    resume_from_checkpoint_path = None
    if args.resume_training:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            resume_from_checkpoint_path = str(ckpt_list[-1])
           
    # Configure trainer options 
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir, # directory for logs and checkpoints
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=resume_from_checkpoint_path,
    )

    args = parser.parse_args()
    
    return args, save_path



def run_main():
    args, save_path = build_args()
    train_test_main(args, save_path)


if __name__ == "__main__":
    run_main()
