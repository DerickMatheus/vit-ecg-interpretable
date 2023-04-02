import torch
import os
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve
# user written
from bcosTransformer import BCosTransformer
from dataloader import CODE2Dataloader
from torch.utils.data import Dataset,ConcatDataset,DataLoader,random_split
import h5py
import utils
from metrics import EcgMetrics
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import datetime


class LitModule(pl.LightningModule):
    def __init__(self, args, nclasses):
        super().__init__()
        self.model = BCosTransformer(dim = args.dim,
                            depth=args.depth,
                            heads=args.heads,
                            dim_head=args.dim_head,
                            nclass=n_classes,
                            seq_len=args.seq_length
                            )
        self.save_hyperparameters()
        
    def training_step(self, batch, batch_id):
        traces, diagnoses = batch
        model_output = self.model(traces)
        loss = F.binary_cross_entropy_with_logits(model_output, diagnoses)
        self.log("train_loss", loss)

        return loss
    
    def test_step(self, batch, batch_id):
        traces, diagnoses = batch
        model_output = self.model(traces)
        loss = F.binary_cross_entropy_with_logits(model_output, diagnoses)
        
        self.log("test_loss", loss)

    def validation_step(self, batch, batch_id):
        traces, diagnoses = batch
        model_output = self.model(traces)
        loss = F.binary_cross_entropy_with_logits(model_output, diagnoses)
        
        self.log("valid_loss", loss)

            
    def configure_optimizers(self):
        #hardcoded!!!
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)
        return optimizer



if __name__ == "__main__":
    import pandas as pd
    import argparse
    from warnings import warn

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict all classes from the raw ecg tracing.')
    
    # Setup arguments
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--path_to_h5', default='/srv/derickmath/aws/h5files/metadata_all_months.h5',
                        help='path to file containing ECG traces for training')
    parser.add_argument('--path_to_csv', default='/srv/derickmath/aws/h5files/classification_data.csv',
                        help='path to csv file containing all diagnoses')
    parser.add_argument('--log_folder', default='logs',
                        help='output folder to log to (default: logs)')
    parser.add_argument('--log_subfolder', default='lightning',
                        help='subfolder to log to e.g. "hyperparameter search" (default: testrun)')
    parser.add_argument('--traces_dset', default='tracings',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--examid_dset', default='exam_id',
                        help='exam id dataset in the hdf5 file.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='prohibit using cuda for computations if available. (default: False)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='checkpoint folder to load previous models. (default: None)')
    
    # training arguments
    parser.add_argument('--epochs', type=int, default=70,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateau (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay in the optimisation. (default: 0)')
    
    # model arguments
    parser.add_argument('--depth', type=int, default=6,
                        help='Number of attentions layers')
    parser.add_argument('--heads', type=int, default=6,
                        help='Number of attention heads for each attention layer')
    parser.add_argument('--dim_head', type=int, default=4096,
                        help='Dimension of each attention head')
    parser.add_argument('--seq_length', type=int, default=1024,
                        help='size (in # of samples) after signal alignment for all traces. If needed traces will be zero padded'
                             'to fit into the given size. (default: 4096)')
    parser.add_argument('--dim', type=int, default=512,
                        help='size (in # of samples) for signals in the embedding space. (default: 512)')
    parser.add_argument('--optim_algo', type=str, default='ADAM',
                        help='Optimation algorithm (default: ADAM).')
    parser.add_argument('--save_intermediary_output', action='store_true',
                        help='When true save the output of the neural network over the intermediary'
                             'epochs.')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    # set seed
    utils.seed_everything(args.seed)

    # Set device
    device = torch.device('cuda:0' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    tqdm.write("Use device: {device:}\n".format(device=device))

    # Generate output folder if needed
    folder = utils.set_output_folder(args.log_folder, args.log_subfolder)

    # Save config file
    utils.save_config(args, folder)

    # =============== Build data loaders =======================================#
    tqdm.write("Building data loaders...")
    
    h5file = h5py.File(args.path_to_h5, 'r')
    train_dataset = CODE2Dataloader(h5file, args.path_to_csv)
    valid_dataset = CODE2Dataloader(h5file, args.path_to_csv)
    train_loader = DataLoader(dataset=train_dataset,batch_size=32,num_workers=4)
    valid_loader = DataLoader(dataset=valid_dataset,batch_size=32,num_workers=4)
#     dset = CODE2Dataset(
#         args.path_to_h5,
#         args.path_to_csv,
#     )
#     # Save dataset (save train and validation outputs and ids)
#     utils.save_dset(dset, folder)
#     # Build dataloader
#     n_valid = sum(dset.val)
#     valid_loader = BatchDataloader(dset, args.batch_size, mask=dset.val)
#     train_loader = BatchDataloader(dset, args.batch_size, mask=dset.train)
    # Get valid expected values
#     validation_truth = dset.outcomes[dset.val]

    tqdm.write("Done!\n")

    # =============== Define model =============================================#
    tqdm.write("Define model...")
    N_LEADS = 8  # the 12 leads (4 are redundant)
    n_classes = 6

    model = LitModule(args, n_classes)
    model.to(device=device)
    tqdm.write("Done!\n")

    # =============== Define loss function =====================================#
    loss_function = torch.nn.BCEWithLogitsLoss()
    
    # =============== Set output directory =====================================#
    
    if args.load_model is None:
        run_folder_name = 'output_' + str(datetime.datetime.now()).replace(":", "_").replace(" ", "_").replace(".", "_")
        folder = os.path.join(os.getcwd(), args.log_folder, args.log_subfolder, run_folder_name)
        if not os.path.exists(folder):
            os.makedirs(folder)        
    # =============== Set trainer and fit the model ============================#
    
    early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="max")
    trainer = pl.Trainer(callbacks=[early_stop_callback],
                         devices=-1,
                         accelerator="gpu",
                         strategy='auto',
                         default_root_dir=folder
                        )
    if args.load_model is None:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    else:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader,ckpt_path=folder + "/model_checkpoint.ckpt")
    h5file.close()
