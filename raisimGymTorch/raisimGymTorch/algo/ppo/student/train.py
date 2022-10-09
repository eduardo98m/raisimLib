import copy
import time
import torch
import argparse
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from Student import StudentEncoder
from datetime import datetime
from torch.optim import lr_scheduler
from TourDataset import TourDataset

def train_model(
        model: torch.nn.Module, 
        batch_size: int,
        device: str,
        checkpoint: str,
        train_data: torch.utils.data.DataLoader,
        valid_data: torch.utils.data.DataLoader,
        myLoss: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler, 
        num_epochs: int,
        max_train_data: int,
        max_valid_data: int
    ) -> torch.nn.Module:
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        since_e = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0
        count = 0
        iter_count = 0

        # Iterate over data.
        print(f'{count} / {max_train_data}', end='\r')
        for data in train_data:
            inputs = data['obs']
            labels = data['label']

            count += inputs.shape[0]

            inputs = inputs.to(device).float()
            labels = labels.to(device).float().reshape((labels.shape[0], -1))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                if torch.isnan(outputs).sum() > 0: 
                    print(f'Nan in output model!')
                    exit(0)
                loss = myLoss(outputs.float(), labels.float())
                if torch.isnan(loss).sum() > 0:
                    print(f'Nan in loss!')
                    exit(0) 
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item()
            iter_count += 1
            print(f'{count} / {max_train_data}', end='\r')
            
            if count >= max_train_data: break

        scheduler.step()

        epoch_loss = running_loss / iter_count
        time_elapsed = time.time() - since_e

        print('Train Loss: {:.4f}'.format(epoch_loss))

        model.eval()  # Set model to eval mode
        running_loss = 0.0
        count = 0
        iter_count = 0

        # Iterate over data.
        print(f'{count} / {max_valid_data}', end='\r')
        for data in valid_data:
            inputs = data['obs']
            labels = data['label']

            count += inputs.shape[0]

            inputs = inputs.to(device).float()
            labels = labels.to(device).float().reshape((labels.shape[0], -1))

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                if torch.isnan(outputs).sum() > 0:
                    print(f'(Valid) Nan in output model!')
                    exit(0) 
                loss = myLoss(outputs.float(), labels.float())
                if torch.isnan(loss).sum() > 0:
                    print(f'(Valid) Nan in loss!')
                    exit(0) 

            # statistics
            running_loss += loss.item()
            iter_count += 1
            print(f'{count} / {max_valid_data}', end='\r')

            if count >= max_valid_data: break

        epoch_loss = running_loss / iter_count
        time_elapsed = time.time() - since_e

        print('Valid Loss: {:.4f}'.format(epoch_loss))


        # Deep copy the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint)

        print(f'Epoch complete in {time_elapsed // 60}m {time_elapsed % 60}s\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
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
    parser.add_argument(
        '-m', '--max-train-data',
        type=int,
        default=100000,
        help='Maxima cantidad de datos a usar en el entrenamiento por epoca',
        metavar='N'
    )
    parser.add_argument(
        '-M', '--max-valid-data',
        type=int,
        default=10000,
        help='Maxima cantidad de datos a usar en el entrenamiento por epoca',
        metavar='N'
    )

    args = parser.parse_args()
    print(args)

    # Cargamos los datos y lo dividimos en entrenamiento y validacion
    data = TourDataset(args.data_folder, args.history_len)
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
    student = StudentEncoder(args.history_len)
    if args.pretrained_model != None:
        student.load_state_dict(torch.load(args.pretrained_model))

    optimizer  = optim.Adam(student.parameters(), lr=0.0001,  weight_decay=1e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    myLoss    = nn.MSELoss()
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student = student.to(device)

    if args.checkpoint_file == None:
        now = datetime.now().strftime('%Y_%m_%d')
        args.checkpoint_file = f'models/{now}.pth'

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
        args.epochs,
        args.max_train_data,
        args.max_valid_data
    )
