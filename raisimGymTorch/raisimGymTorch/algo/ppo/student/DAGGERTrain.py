import copy
import time
import torch
import argparse
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from Student import Student, StudentEncoder
from datetime import datetime
from torch.optim import lr_scheduler
from DAGGERDataset import DAGGERDataset

def train_model_epoch(
        model: Student, 
        batch_size: int,
        device: str,
        checkpoint: str,
        train_data: torch.utils.data.DataLoader,
        valid_data: torch.utils.data.DataLoader,
        myLoss: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler, 
        best_loss: float,
    ) -> torch.nn.Module:


    since_e = time.time()
    print('-' * 10)

    model.train()  # Set model to training mode
    running_loss = 0.0
    count = 0

    # Iterate over data.
    total_train_data = len(train_data) * batch_size
    print(f'{count} / {total_train_data}', end='\r')
    for data in train_data:
        obs_inputs = data['obs']
        H_inputs = data['H']
        labels = data['label']
        actions = data['action']

        count += obs_inputs.shape[0]

        obs_inputs = obs_inputs.to(device).float()
        H_inputs = H_inputs.to(device).float()
        labels = labels.to(device).float().reshape((labels.shape[0], -1))
        actions = actions.to(device).float().reshape((actions.shape[0], -1))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            encoder_outputs, regressor_outputs = model.forward_encoder(
                obs_inputs,
                H_inputs
            )
            if torch.isnan(regressor_outputs).sum() > 0: 
                print(f'Nan in output model!')
                exit(0)
            # TODO: Use the actions in the loss calculation
            loss = myLoss(
                encoder_outputs.float(), 
                labels.float(), 
                regressor_outputs.float(),
                actions.float()
            )
            if torch.isnan(loss).sum() > 0:
                print(f'Nan in loss!')
                exit(0) 
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item()
        print(f'{count} / {total_train_data}', end='\r')

    scheduler.step()

    epoch_loss = running_loss / len(train_data)
    time_elapsed = time.time() - since_e

    print('Train Loss: {:.4f}'.format(epoch_loss))

    model.eval()  # Set model to eval mode
    running_loss = 0.0
    count = 0

    # Iterate over data.
    total_valid_data = len(valid_data) * batch_size
    print(f'{count} / {total_valid_data}', end='\r')
    for data in valid_data:
        obs_inputs = data['obs']
        H_inputs = data['H']
        labels = data['label']
        actions = data['action']

        count += obs_inputs.shape[0]

        obs_inputs = obs_inputs.to(device).float()
        H_inputs = H_inputs.to(device).float()
        labels = labels.to(device).float().reshape((labels.shape[0], -1))
        actions = actions.to(device).float().reshape((actions.shape[0], -1))

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            encoder_outputs, regressor_outputs = model.forward_encoder(
                obs_inputs,
                H_inputs
            )
            if torch.isnan(regressor_outputs).sum() > 0: 
                print(f'Nan in output model!')
                exit(0)
            # TODO: Use the actions in the loss calculation
            loss = myLoss(
                encoder_outputs.float(), 
                labels.float(), 
                regressor_outputs.float(),
                actions.float()
            )
            if torch.isnan(loss).sum() > 0:
                print(f'Nan in loss!')
                exit(0) 

        # statistics
        running_loss += loss.item()
        print(f'{count} / {total_valid_data}', end='\r')

    epoch_loss = running_loss / len(valid_data)
    time_elapsed = time.time() - since_e

    print('Valid Loss: {:.4f}'.format(epoch_loss))

    if best_loss > epoch_loss:
        print("Model improved from {:.4f} to {:.4f}".format(best_loss, epoch_loss))
        print("Model saved")
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, checkpoint)

    print(f'Epoch complete in {time_elapsed // 60}m {time_elapsed % 60}s\n')

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entena un modelo Estudiante.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-d', '--data-folder',
        type=str,
        default='data',
        help='Directorio que contiene todas las runs.',
        metavar='PATH'
    )
    parser.add_argument(
        '-H', '--history-len',
        type=int,
        default=60,
        help='Longitud de las secuencias que se pasara al estudiante.',
        metavar='LEN'
    )
    parser.add_argument(
        '-t', '--train-split',
        type=float,
        default=0.75,
        help='Proporcion de datos que se usara para entrenamiento',
        metavar='SPLIT'
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=32,
        help='Batch size del entrenamiento.',
        metavar='SIZE'
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=10,
        help='Numero de epocas que durara el entrenamiento.',
        metavar='EPOCHS'
    )
    parser.add_argument(
        '-c', '--checkpoint-file',
        type=str,
        default=None,
        help='Archivo donde se guardara el modelo entrenado.',
        metavar='FILE'
    )
    parser.add_argument(
        '-p', '--pretrained-model',
        type=str,
        default=None,
        help='Cargar los pesos de un modelo pre-entrenado.',
        metavar='FILE'
    )
    parser.add_argument(
        '-D', '--dropout',
        type=float,
        default=0.1,
        help='Dropout que se aplicara durante el entrenamiento',
        metavar='DROPOUT'
    )

    args = parser.parse_args()
    print(args)

    # Cargamos los datos y lo dividimos en entrenamiento y validacion
    data = DAGGERDataset(args.data_folder, begin_H, end_H, args.history_len)
    train_len = int(len(data) * args.train_split)
    valid_len = len(data) - train_len 
    train_data, valid_data = torch.utils.data.random_split(
        data, 
        [train_len, valid_len]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, 
        batch_size=args.batch_size, 
        shuffle=True
    )

    # Cargamos el modelo
    encoder = StudentEncoder(args.history_len)
    student = Student(teacher, encoder)
    if args.pretrained_model != None:
        student.load_state_dict(torch.load(args.pretrained_model))

    optimizer  = optim.Adam(student.parameters(), lr=0.001,  weight_decay=1e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    myLoss    = nn.MSELoss() # AQUI HAY QUE CAMBIAR LA FUNCION DE PERDIDA
    device    = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    student = student.to(device)

    if args.checkpoint_file == None:
        now = datetime.now().strftime('%Y_%m_%d')
        args.checkpoint_file = f'models/{now}.pth'

    """
    train_model(
        student,
        args.batch_size,
        device,
        args.checkpoint_file,
        train_dataloader,
        valid_dataloader,
        myLoss,
        optimizer,
        scheduler,
        args.epochs
    )
    """
