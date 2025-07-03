import torch
import torch.nn as nn
import math
from transformer import TransformerEncoderLayer, TransformerEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        max_len = max(5000, seq_len)
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
            
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0)
        

class MultitaskTransformerModel(nn.Module):
    def __init__(self, task_type, device, nclasses, seq_len, batch, input_size, emb_size, nhead, nhid, nhid_tar, nhid_task, nlayers, dropout=0.1):
        super(MultitaskTransformerModel, self).__init__()
        
        self.trunk_net = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.BatchNorm1d(batch),
            PositionalEncoding(seq_len, emb_size, dropout),
            nn.BatchNorm1d(batch)
        )
        
        encoder_layer = TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers, device)
        
        self.batch_norm = nn.BatchNorm1d(batch)
        
        self.tar_net = nn.Sequential(
            nn.Linear(emb_size, nhid_tar),
            nn.BatchNorm1d(batch),
            nn.Linear(nhid_tar, nhid_tar),
            nn.BatchNorm1d(batch),
            nn.Linear(nhid_tar, input_size)
        )

        if task_type == 'classification':
            self.class_net = nn.Sequential(
                nn.Linear(emb_size, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Dropout(0.3),
                nn.Linear(nhid_task, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Dropout(0.3),
                nn.Linear(nhid_task, nclasses)
            ) 
        else:
            self.reg_net = nn.Sequential(
                nn.Linear(emb_size, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Linear(nhid_task, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Linear(nhid_task, 1)
            )
        
    
    def forward(self, x, task_type):
        x = self.trunk_net(x.permute(1, 0, 2))
        x, attn = self.transformer_encoder(x)
        x = self.batch_norm(x)
        
        if task_type == 'reconstruction':
            output = self.tar_net(x).permute(1, 0, 2)
        elif task_type == 'classification':
            output = self.class_net(x[-1])
        elif task_type == 'regression':
            output = self.reg_net(x[-1])
        
        return output, attn
    
    
def main():
    device = 'cuda:0'
    lr, dropout = 0.01, 0.01
    nclasses, seq_len, batch, input_size = 12, 5, 11, 10
    emb_size, nhid, nhead, nlayers = 32, 128, 2, 3
    nhid_tar, nhid_task = 128, 128
    task_type = 'regression'
    
    model = MultitaskTransformerModel(task_type, device, nclasses, seq_len, batch, input_size, emb_size, nhead, nhid, nhid_tar, nhid_task, nlayers, dropout = 0.1).to(device)
    
    x = torch.randn(batch, seq_len, input_size) * 50
    x = torch.as_tensor(x).float()
    print(x.shape)
    
    output_tar, attn_tar = model(torch.as_tensor(x, device = device), 'reconstruction')
    output_task, attn_task = model(torch.as_tensor(x, device = device), task_type)
    
    print(output_tar.shape)
    print(attn_tar.shape)
    
    print(output_task.shape)
    print(attn_task.shape)
    
    print(output_task)
    


if __name__ == '__main__':
    main()