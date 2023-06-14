import cv2
import torch
import numpy as np

# Character dictionary and reverse character dictionary created
all_chars = [chr(x) for x in range(32,127)]
char_dict = {i+4:x for i,x in enumerate(all_chars)}
char_dict[0] = 'unk'
char_dict[1] = 'pad'
char_dict[2] = 'bos'
char_dict[3] = 'eos'
rev_char = {y:x for x,y in char_dict.items()}

# Collate function for Dataloader to utilize
def generate_batch(data_batch,w,h):
    
    # initializations
    im_list, txt_tokens = [], []
    trg_pad_mask = []
    max_idx = -1
    max_len = 0

    # finding out max length of sequence in batch
    for _,txt_path in data_batch:
        with open(txt_path)as f:
            ann = f.readline()
            
        if len(ann)>max_len:
            max_len = len(ann)
            
    max_len+=3
    
    # Iterating over entries in batch
    for im_path,txt_path in data_batch:
        
        entry_txt_tokens = []
        
        # Reading image and resizing, also BGR to RGB
        im = cv2.cvtColor(cv2.resize(cv2.imread(im_path),(w,h)),cv2.COLOR_BGR2RGB)

        # Reading transcription for corresponding image
        with open(txt_path)as f:
            ann = f.readline()

            # converting transcription from ascii text to numerical representation
            chars = [rev_char[x] for x in list(ann)]
            
        num_chars = len(chars)
        num_pad = max_len - num_chars - 2
        
        if num_chars+1 > max_idx:
            max_idx = num_chars+1

        # Mask for entry. With padding if not longest entry in batch
        temp_mask = []
        temp_mask.extend([False for _ in range(num_chars+2)])
        temp_mask.extend([True for _ in range(num_pad)])
        trg_pad_mask.append(temp_mask)

        # Token sequence for transcription. 2 as bos, 3 as eos and 1 as pad
        entry_txt_tokens = [2]+chars+[3]+[1 for _ in range(num_pad)]
        
        txt_tokens.append(entry_txt_tokens)
        
        im_list.append(im)
        
    # Torch tensor created and perrmuted to B,C,H,W format
    im_list = torch.tensor(np.array(im_list)).permute((0,3,1,2))
    
    txt_tokens = torch.tensor(txt_tokens)[:,:max_idx+1].permute((1,0))
    
    trg_pad_mask = torch.tensor(trg_pad_mask, dtype=torch.bool)[:,:max_idx]
    
    return im_list,trg_pad_mask,txt_tokens