import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            ni, nf, ks,
            stride=stride,
            padding=padding,
            dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(
            nf, nf, ks, 
            stride=stride, 
            padding=padding, 
            dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, 
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(ni,nf,1) if ni != nf else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None: 
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class StudentEncoder(nn.Module):
    def __init__(self, 
                seq_len: int=60, 
                out_tcn_len: int=49, 
                dropout=0., 
                old_version = False):
        super().__init__()

        self.seq_len = seq_len 
        kernel_size = 5
        strides     = [1, 1, 1, 1, 1, 1]
        dilatations = [1, 1, 2, 1, 4, 1]
        paddings    = [4 * d for d in dilatations]
        
        tcn_modules = []
        for i in range(len(strides)):
            tcn_modules.append(TemporalBlock(
                seq_len, 
                seq_len, 
                kernel_size, 
                strides[i], 
                dilatations[i], 
                paddings[i],
                dropout=dropout
            ))
            tcn_modules.append(nn.LeakyReLU())
        self.tcn = nn.Sequential(*tcn_modules)
        self.fc = nn.Linear(out_tcn_len, 64)
        
        if old_version:
            self.activation = nn.Identity()
        else:
            self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.tcn(x)[:,0,:]
        return self.activation(self.fc(x))
        #return self.fc(x)

class Student(nn.Module):
    def __init__(self, teacher, student_encoder, disable_grad_on_classifier = True, disable_grad_on_TCN = False):
        super().__init__()

        self.encoder = student_encoder
        self.classifier = teacher.classifier

        if disable_grad_on_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = False
        
        if disable_grad_on_TCN:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, obs, history):
        x = self.encoder(history)
        x = torch.cat((x, obs), 1)

        return self.classifier(x)

    def forward_encoder(self, obs, history):
        encoder_output = self.encoder(history)
        x = torch.cat((encoder_output, obs), 1)

        return encoder_output, self.classifier(x)
    
class StudentRegressor(nn.Module):
    def __init__(self, classifier, encoder_output_dim, non_priv_dim, output_dim):
        super().__init__()
        self.classifier = classifier

        self.input_shape = [encoder_output_dim + non_priv_dim]
        self.output_shape = [output_dim]
    
    def forward(self, x):

        return self.classifier(x)