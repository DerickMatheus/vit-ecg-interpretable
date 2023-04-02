import torch
import os
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_curve
# user written
from bcosTransformer import BCosTransformer
from dataloader import CODE2Dataset, BatchDataloader
import utils
from metrics import EcgMetrics


def train(ep, dataload):
    # model to training
    model.train()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0  # accumulated number of data points
    # progress bar def
    train_pbar_desc = "Training Epoch {epoch:2d}".format(epoch=ep)
    train_pbar = tqdm(dataload, desc=train_pbar_desc, leave=False, initial=0, position=0)
    # training loop
    for traces, diagnoses in train_pbar:
        # data to device
        traces, diagnoses = traces.to(device), diagnoses.to(device)  # use cuda if available

        # Reinitialize grad
        model.zero_grad()

        # Forward pass
        model_output = model(traces[:,1,:])
        loss = loss_function(model_output, diagnoses)

        # Backward pass
        loss.backward()

        # Optimize
        optimiser.step()

        # Update accumulated values
        total_loss += loss.detach().cpu().numpy()
        n_entries += len(traces)

        # Update progress bar
        train_pbar.set_postfix({'loss': total_loss / n_entries})
    train_pbar.close()
    return total_loss / n_entries


def eval(ep, dataload, n_valid, n_classes):
    # model to validation
    model.eval()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0  # accumulated number of data points
    eval_pred = np.zeros((n_valid, n_classes))
    # end = dataload.start_idx
    end = 0
    # progress bar def
    eval_pbar_desc = "Evaluation Epoch {epoch:2d}".format(epoch=ep)
    eval_pbar = tqdm(dataload, desc=eval_pbar_desc, leave=False, initial=0, position=0)
    # evaluation loop
    for traces_cpu, diagnoses_cpu in eval_pbar:
        # data to device
        traces, diagnoses = traces_cpu.to(device), diagnoses_cpu.to(device)
        start = end
        with torch.no_grad():
            # Forward pass
            model_output = model(traces[:,1,:])
            loss = loss_function(model_output, diagnoses)

            # store output
            end = min(start + len(model_output), eval_pred.shape[0])
            eval_pred[start:end] = torch.nn.Sigmoid()(model_output).detach().cpu().numpy()

            # Update accumulated values
            total_loss += loss.detach().cpu().numpy()
            n_entries += len(traces)

            # Update progress bar
            eval_pbar.set_postfix({'loss': total_loss / n_entries})
    eval_pbar.close()
    return total_loss / n_entries, eval_pred


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
    parser.add_argument('--path_to_h5', default='../data/examples.h5',
                        help='path to file containing ECG traces for training')
    parser.add_argument('--path_to_csv', default='../data/classification_data.csv',
                        help='path to csv file containing all diagnoses')
    parser.add_argument('--log_folder', default='logs',
                        help='output folder to log to (default: logs)')
    parser.add_argument('--log_subfolder', default='testrun',
                        help='subfolder to log to e.g. "hyperparameter search" (default: testrun)')
    parser.add_argument('--traces_dset', default='tracings',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--examid_dset', default='exam_id',
                        help='exam id dataset in the hdf5 file.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='prohibit using cuda for computations if available. (default: False)')
    
    
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
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of attentions layers')
    parser.add_argument('--heads', type=int, default=12,
                        help='Number of attention heads for each attention layer')
    parser.add_argument('--dim_head', type=int, default=4096,
                        help='Dimension of each attention head')
    parser.add_argument('--seq_length', type=int, default=4096,
                        help='size (in # of samples) for all traces. If needed traces will be zero padded'
                             'to fit into the given size. (default: 4096)')
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
    dset = CODE2Dataset(
        args.path_to_h5,
        args.path_to_csv,
    )
    # Save dataset (save train and validation outputs and ids)
    utils.save_dset(dset, folder)
    # Build dataloader
    n_valid = sum(dset.val)
    valid_loader = BatchDataloader(dset, args.batch_size, mask=dset.val)
    train_loader = BatchDataloader(dset, args.batch_size, mask=dset.train)
    # Get valid expected values
    validation_truth = dset.outcomes[dset.val]

    tqdm.write("Done!\n")

    # =============== Define model =============================================#
    tqdm.write("Define model...")
    N_LEADS = 8  # the 12 leads (4 are redundant)
    n_classes = len(dset.outcomes.keys())

    model = BCosTransformer(dim = args.seq_length,
                            depth=args.depth,
                            heads=args.heads,
                            dim_head=args.dim_head,
                            nclass=n_classes,
                            )
    model.to(device=device)
    tqdm.write("Done!\n")

    # =============== Define loss function =====================================#
    loss_function = torch.nn.BCEWithLogitsLoss()

    # =============== Define optimiser =========================================#
    tqdm.write("Define optimiser...")
    if(args.optim_algo == 'SGD'):
        optimiser = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tqdm.write("Done!\n")

    # =============== Define lr scheduler ======================================#
    tqdm.write("Define scheduler...")
    # learning rate scheduler based on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                     patience=args.patience,
                                                     min_lr=args.lr_factor * args.min_lr,
                                                     factor=args.lr_factor)
    tqdm.write("Done!\n")

    # =============== Set up Logger ============================================#
    tqdm.write("Set up logger and metrics...")
    metrics = EcgMetrics(n_classes)
    logger = utils.Logger(folder, metrics.metric_names)
    tqdm.write("Done!\n")

    # =============== Train model ==============================================#
    tqdm.write("Training...")
    best_loss = np.Inf

    # create data frame to store the results in
    logger.init_history(columns=['epoch', 'train_loss', 'valid_loss', 'lr'] + metrics.metric_names)

    # loop over epochs
    for ep in range(1, args.epochs + 1):
        train_loss = train(ep, train_loader)
        valid_loss, valid_pred = eval(ep, valid_loader, n_valid, n_classes)

        # save validation predictions
        if args.save_intermediary_output:
            logger.log_valid_pred(ep, valid_pred)

        # compute validation metrics
        valid_metrics = metrics.compute_metrics(valid_pred, validation_truth.values)

        # save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),  # model parameters
                        'valid_loss': valid_loss,
                        'optimiser': optimiser.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_loss = valid_loss
            # statement
            model_save_state = "Best model -> saved"
        else:
            model_save_state = ""

        # Get learning rate
        for param_group in optimiser.param_groups:
            learning_rate = param_group["lr"]

        # Interrupt for minimum learning rate
        if learning_rate < args.min_lr:
            tqdm.write("Stopped since learning rate minimum is reached!")
            break

        # Print message
        tqdm.write('Epoch {epoch:2d}: \t'
                   'Train Loss {train_loss:.6f} \t'
                   'Valid Loss {valid_loss:.6f} \t'
                   'Learning Rate {lr:.7f}\t'
                   '{model_save}'
                   .format(epoch=ep,
                           train_loss=train_loss,
                           valid_loss=valid_loss,
                           lr=learning_rate,
                           model_save=model_save_state)
                   )

        # data to log
        log_data = {"epoch": ep,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "lr": learning_rate}
        log_data.update(valid_metrics)
        # log data
        logger.log_all(log_data, ep)
        # Save history
        logger.save_history(log_data)

        # Update learning rate with lr-scheduler
        scheduler.step(valid_loss)
    # add true validation output to the log and save


    tqdm.write("Done!\n")
