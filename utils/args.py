# Dataset setup arguments

# Args for filtering samples
exclude_ascii = [chr(0),chr(9),chr(10)]
max_HbyW = 5
max_WbyH = 10
max_height = 250
max_width = 400
min_height = 6
min_width = 7
max_str_len = 18

# Images with ratio between these two will be consider square
max_HbyW_for_sq = 1.5 # Images with H/W ratio greater than this will be considered tall
max_WbyH_for_sq = 1.5 # Images with W/H ratio greater than this will be considered wide



# Training arguments

BATCH_SIZE = 160
DEVICE = 'cuda'
EPOCH = 300
file_dest = 'ckpts/'

wide_res = (96,288) #(144,432)
square_res = (160,160) #(240,240)
tall_res = (224,112) #(352,176)

resolution_dist = [0.05515061891144885,0.31132706679868344,0.6335223142898677]
