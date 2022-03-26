import argparse
import time
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.optim import Adam

from models.GAT import GAT
from utils.ppi_loader import load_graph_data
from utils.constants import *
import utils.utils as utils



# Simple decorator function so that no need to pass arguments that don't change from epoch to epoch
def get_main_loop(config, gat, sigmoid_cross_entropy_loss, optimizer, patience_period, time_start):

    device = next(gat.parameters()).device  # fetch the device info from the model instead of passing it as a param

    def main_loop(phase, data_loader, epoch=0):
        global BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT, writer

        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        # Iterate over batches of graph data (2 graphs per batch was used in the original paper for the PPI dataset)
        # Merge them into a single graph with 2 connected components, that's the main idea.
        for batch_idx, (node_features, gt_node_labels, edge_index) in enumerate(data_loader):
            edge_index = edge_index.to(device)
            node_features = node_features.to(device)
            gt_node_labels = gt_node_labels.to(device)

            # Pack data into tuples because GAT uses nn.Sequential which expects this format
            graph_data = (node_features, edge_index)

            # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
            # shape = (N, C) where N is the number of nodes in the batch and C is the number of classes (121 for PPI)
            nodes_unnormalized_scores = gat(graph_data)[0]

            # First apply a sigmoid and the apply a binary cross entropy
            loss = sigmoid_cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)

            if phase == LoopPhase.TRAIN:
                optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
                loss.backward()  # compute the gradients for every trainable weight in the computational graph
                optimizer.step()  # apply the gradients to weights

            # Calculate the main metric - micro F1

            # Convert unnormalized scores into predictions. Explanation:
            # If the unnormalized score is bigger than 0 that means that sigmoid would have a value higher than 0.5
            # (by sigmoid's definition) and thus we have predicted 1 for that label otherwise we have predicted 0.
            pred = (nodes_unnormalized_scores > 0).float().cpu().numpy()
            gt = gt_node_labels.cpu().numpy()
            micro_f1 = f1_score(gt, pred, average='micro')

            # Logging

            global_step = len(data_loader) * epoch + batch_idx
            if phase == LoopPhase.TRAIN:
                # Log metrics
                if config['enable_tensorboard']:
                    writer.add_scalar('training_loss', loss.item(), global_step)
                    writer.add_scalar('training_micro_f1', micro_f1, global_step)

                # Log to console
                if config['console_log_freq'] is not None and batch_idx % config['console_log_freq'] == 0:
                    print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] |'
                          f' epoch={epoch + 1} | batch={batch_idx + 1} | train micro-F1={micro_f1}.')

                # Save model checkpoint
                if config['checkpoint_freq'] is not None and (epoch + 1) % config['checkpoint_freq'] == 0 and batch_idx == 0:
                    ckpt_model_name = f'gat_PPI_ckpt_epoch_{epoch + 1}.pth'
                    config['test_perf'] = -1  # test perf not calculated yet, note: perf means main metric micro-F1 here
                    torch.save(gat.state_dict(), os.path.join(config['ckpt_path'], ckpt_model_name))

            elif phase == LoopPhase.VAL:
                # Log metrics
                if config['enable_tensorboard']:
                    writer.add_scalar('val_loss', loss.item(), global_step)
                    writer.add_scalar('val_micro_f1', micro_f1, global_step)

                # Log to console
                if config['console_log_freq'] is not None and batch_idx % config['console_log_freq'] == 0:
                    print(f'GAT validation: time elapsed= {(time.time() - time_start):.2f} [s] |'
                          f' epoch={epoch + 1} | batch={batch_idx + 1} | val micro-F1={micro_f1}')

                # The "patience" logic - should we break out from the training loop? If either validation micro-F1
                # keeps going up or the val loss keeps going down we won't stop
                if micro_f1 > BEST_VAL_PERF or loss.item() < BEST_VAL_LOSS:
                    BEST_VAL_PERF = max(micro_f1, BEST_VAL_PERF)  # keep track of the best validation micro_f1 so far
                    BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)  # and the minimal loss
                    PATIENCE_CNT = 0  # reset the counter every time we encounter new best micro_f1
                else:
                    PATIENCE_CNT += 1  # otherwise keep counting

                if PATIENCE_CNT >= patience_period:
                    raise Exception('Stopping the training, the universe has no more patience for this training.')

            else:
                return micro_f1  # in the case of test phase we just report back the test micro_f1

    return main_loop  # return the decorated function


def train_gat_ppi(config):
    """
    Using dataloaders since we're dealing with an inductive setting - multiple graphs per batch
    Doing multi-class classification (BCEWithLogitsLoss) and reporting micro-F1 instead of accuracy
    """
    global BEST_VAL_PERF, BEST_VAL_LOSS

    if config['cuda_idx'] is not None:
        device = torch.device(f"cuda:{config['cuda_idx']}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loader_train, data_loader_val, data_loader_test = load_graph_data(config, device)

    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    os.makedirs(config['ckpt_path'], exist_ok=True)
    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        config,
        gat,
        loss_fn,
        optimizer,
        config['patience_period'],
        time.time())

    BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # reset vars used for early stopping

    for epoch in range(config['num_of_epochs']):
        # Training loop
        main_loop(phase=LoopPhase.TRAIN, data_loader=data_loader_train, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                main_loop(phase=LoopPhase.VAL, data_loader=data_loader_val, epoch=epoch)
            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break  # break out from the training loop

    # Potentially test the model
    # Don't overfit to the test dataset - only when you've fine-tuned your model on the validation dataset should you
    # report your final loss and micro-F1 on the test dataset. Friends don't let friends overfit to the test data. <3
    if config['should_test']:
        micro_f1 = main_loop(phase=LoopPhase.TEST, data_loader=data_loader_test)
        config['test_perf'] = micro_f1

        print('*' * 50)
        print(f'Test micro-F1 = {micro_f1}')
    else:
        config['test_perf'] = -1

    # Save the latest GAT in the binaries directory
    torch.save(gat.state_dict(), os.path.join(config['ckpt_path'], 'gat_PPI_final.pth'))


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=200)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=100)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=0)
    parser.add_argument("--should_test", action='store_true', help='should test the model on the test dataset? (no by default)')
    parser.add_argument("--cuda_idx", type=int, default=None, help="cuda indice, None for 0")

    # Dataset related (note: we need the dataset name for metadata and related stuff, and not for picking the dataset)
    parser.add_argument("--ppi_path", type=str, default='./data/ppi', help='path to ppi metadata')
    parser.add_argument("--batch_size", type=int, help='number of graphs in a batch', default=2)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq (None for no logging)", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq (None for no logging)", default=5)
    parser.add_argument("--ckpt_path", type=str, default='./weights', help="path to save checkpoint")
    args = parser.parse_args()

    # I'm leaving the hyperparam values as reported in the paper, but I experimented a bit and the comments suggest
    # how you can make GAT achieve an even higher micro-F1 or make it smaller
    gat_config = {
        # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_of_layers": 3,  # PPI has got 42% of nodes with all 0 features - that's why 3 layers are useful
        "num_heads_per_layer": [4, 4, 6],  # other values may give even better results from the reported ones
        "num_features_per_layer": [50, 256, 256, 121],  # 64 would also give ~0.975 uF1!
        "add_skip_connection": True,  # skip connection is very important! (keep it otherwise micro-F1 is almost 0)
        "bias": True,  # bias doesn't matter that much
        "dropout": 0.0,  # dropout hurts the performance (best to keep it at 0)
    }

    # Wrap training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    # Add additional config information
    training_config.update(gat_config)

    return training_config


if __name__ == '__main__':

    # Train the graph attention network (GAT)
    train_gat_ppi(get_training_args())
