import torch
import torch.nn as nn
import torchvision
from torch import Tensor
import math
import numpy as np

# 1D positional encoding
class PositionalEncoding(nn.Module):

    def __init__(self, emb_size: int, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return (token_embedding + self.pos_embedding[:token_embedding.size(0),:])

# Generates n dimentional embedding for a token
class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Main model class
class Network(nn.Module):

    # Generates 2D positional embedding
    def pos_embed_2d(self,shape_2d, dims):
    
        y, x = shape_2d
        quart = dims // 4

        dim_pos = torch.arange((quart))
        dim_vals = 10000**((4*dim_pos)/dims)

        x_mat = torch.stack([torch.ones(y,quart)*i for i in range(x)],dim=1)
        y_mat = torch.stack([torch.ones(x,quart)*i for i in range(y)],dim=0)

        mat_a = torch.sin(x_mat/dim_vals)
        mat_b = torch.cos(x_mat/dim_vals)
        mat_c = torch.sin(y_mat/dim_vals)
        mat_d = torch.cos(y_mat/dim_vals)

        leaved_first_half = torch.stack((mat_a,mat_b), dim=-1).view(y,x,-1)
        leaved_second_half = torch.stack((mat_c,mat_d), dim=-1).view(y,x,-1)

        res = torch.cat((leaved_first_half,leaved_second_half),dim=-1)

        return res

    # Generates mask for decoder to use
    def generate_square_subsequent_mask(self,sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    
    def __init__(self,device):
        
        super().__init__()

        # Dimension of one embedding
        self.dims = 128

        # Vocabulary size
        self.vocab_len = 99

        # Device o be used
        self.device = device

        # height and width of input after passing through CNN backbone, will be updated at forward pass
        self.h = 0
        self.w = 0
        
        # CNN backbone to generate patch embeddings from image. Excluding last downsample to get patch embeddings for 16x16 patches
        self.backbone = nn.Sequential(*(list(torchvision.models.mobilenet_v3_small(weights='DEFAULT').children())[:-2][0][:-4]))
        
        # 1x1 conv layer to alter channels to required dimensions
        self.conv = nn.Conv2d(in_channels=48, out_channels=self.dims, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.dims)

        # transformer encoders
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dims, nhead=4,dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3).to(device)
        
        # transformer decoders
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.dims, nhead=4,dim_feedforward=512)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3).to(device)
        
        # 1D positional encoding for character sequence
        self.positional_encoding = PositionalEncoding(self.dims).to(device)
        
        # Convertes token to embedding. 4 special tokens + 95 ASCII characters
        self.tok_emb = TokenEmbedding(self.vocab_len, self.dims)

        # Src mask and pad maask, will be updated at forward pass
        self.src_mask = None
        self.src_pad_mask = None

        # Final linear layer to convert to vocab dimensional output
        self.fc = nn.Linear(in_features=self.dims, out_features=self.vocab_len)
    
    def forward(self,x,y,trg_pad_mask):

        # Height and width calculated
        self.h = int(np.ceil(x.shape[2]/16))
        self.w = int(np.ceil(x.shape[3]/16))
        
        # input passed through backbone and 1x1 conv
        t1 = self.backbone(x)
        t1 = self.conv(t1)
        t1 = self.bn(t1)
       
        # Dimensions permuted to have embeddings at the end
        t1 = t1.permute(0,2,3,1)

        # 2D positional encoding added to CNN output
        inp = t1 + self.pos_embed_2d((self.h,self.w),self.dims).to(self.device)
        
        # Unrolled to remove width height info, as 2D positional info is now present
        inp = inp.reshape(-1,self.h*self.w,self.dims) 

        # Dimensions permuted to have number of embeddings first, and batch size second. (Normal transformer format)
        inp = inp.permute((1,0,2))
        
        # Source mask and pad mask generated
        self.src_mask = torch.zeros((self.h*self.w,self.h*self.w), dtype=torch.bool).to(self.device)
        self.src_pad_mask = torch.zeros((x.shape[0],self.h*self.w), dtype=torch.bool).to(self.device)

        # Output tokenized and 1D positional embedding added
        y = self.positional_encoding(self.tok_emb(y))
        len_y = y.shape[0]
        y_mask = self.generate_square_subsequent_mask(len_y)
        
        # Activations passed through encoder and decoder with appropriate masks and pad masks
        memory = self.transformer_encoder(inp, self.src_mask, self.src_pad_mask[:trg_pad_mask.shape[0],:])
        outs = self.transformer_decoder(y, memory, y_mask, None, trg_pad_mask, self.src_pad_mask[:trg_pad_mask.shape[0],:])

        # Output turned into vocab length dimensional
        return self.fc(outs)
    
    def encode(self, src: Tensor):

        t1 = self.backbone(src)
        t1 = self.conv(t1)
        t1 = self.bn(t1)
        t1 = t1.permute(0,2,3,1)

        inp = self.pos_mat + t1
        inp = inp.reshape(-1,self.h*self.w,self.dims) 
        inp = inp.permute((1,0,2))

        return self.transformer_encoder(inp, self.src_mask)


    def decode(self, memory: Tensor):

        batch_bool = torch.zeros(memory.shape[1], dtype=torch.bool).to(self.device)

        ys = torch.ones(1,memory.shape[1]).fill_(2).type(torch.long).to(self.device)
        tmp_y = self.tok_emb(ys).to(self.device)
        tgt_mask = (self.generate_square_subsequent_mask(ys.shape[0]).type(torch.bool)).to(self.device)

        for _ in range(32):

            out = self.transformer_decoder(self.positional_encoding(tmp_y), memory, tgt_mask)
            out = out.transpose(0, 1)

            prob = self.fc(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.reshape(1,-1)
            next_word[batch_bool.view(1,memory.shape[1])] = 1

            ys = torch.vstack((ys,next_word)).type(torch.long)
            batch_bool = torch.logical_or(batch_bool,(next_word.view(-1)==3))

            if torch.sum(batch_bool) == memory.shape[1]:
                break

            tmp_y = self.tok_emb(ys).to(self.device)
            tgt_mask = (self.generate_square_subsequent_mask(ys.shape[0]).type(torch.bool)).to(self.device)

        return ys.transpose(0,1)
