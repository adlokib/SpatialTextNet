import cv2
import torch
import numpy as np

chars = [chr(x) for x in range(32,127)]
char_dict = {i+4:x for i,x in enumerate(chars)}
char_dict[0] = 'unk'
char_dict[1] = 'pad'
char_dict[2] = 'bos'
char_dict[3] = 'eos'
rev_char = {y:x for x,y in char_dict.items()}

def generate_batch(data_batch,w,h):
    
    im_list, txt_tokens = [], []
    
    trg_pad_mask = []
    
    max_idx = -1
    
    max_len = 0
    for _,txt_path in data_batch:
        with open(txt_path)as f:
            ann = f.readline()
            
        if len(ann)>max_len:
            max_len = len(ann)
            
    max_len+=3
    
    for im_path,txt_path in data_batch:
        
        entry_txt_tokens = []
        
        im = cv2.cvtColor(cv2.resize(cv2.imread(im_path),(w,h)),cv2.COLOR_BGR2RGB)

        with open(txt_path)as f:
            ann = f.readline()
            chars = [rev_char[x] for x in list(ann)]
            
        num_chars = len(chars)
        num_pad = max_len - num_chars - 2
        
        if num_chars+1 > max_idx:
            max_idx = num_chars+1

        temp_mask = []
        temp_mask.extend([False for _ in range(num_chars+2)])
        temp_mask.extend([True for _ in range(num_pad)])
        
        trg_pad_mask.append(temp_mask)
        entry_txt_tokens = [2]+chars+[3]+[1 for _ in range(num_pad)]
        
        txt_tokens.append(entry_txt_tokens)
        
        im_list.append(im)
        
    im_list = torch.tensor(np.array(im_list)).permute((0,3,1,2))
    
    txt_tokens = torch.tensor(txt_tokens)[:,:max_idx+1].permute((1,0))
    
    trg_pad_mask = torch.tensor(trg_pad_mask, dtype=torch.bool)[:,:max_idx]
    
    return im_list,trg_pad_mask,txt_tokens