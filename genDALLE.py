import torch
from dalle_pytorch import DiscreteVAE, DALLE
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.utils import save_image
import time
import sys

# vae

load_epoch = 390
vaename = "vae-cdim256"

# general

imgSize = 256
batchSize = 12
n_epochs = 100
log_interval = 10
lr = 2e-5

#dalle

dalle_epoch = 220
#loadfn = ""
#start_epoch = 0
name = "vae-cdim256"
loadfn = "./models/dalle_"+name+"-"+str(dalle_epoch)+".pth"   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tf = transforms.Compose([
  #transforms.Resize(imgSize),
  #transforms.RandomHorizontalFlip(),
  #transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #(0.267, 0.233, 0.234))
  ])

vae = DiscreteVAE(
    image_size = 256,
    num_layers = 3,
    num_tokens = 2048,
    codebook_dim = 256,
    hidden_dim = 128,
    temperature = 0.9
)

# load pretrained vae

vae_dict = torch.load("./models/"+vaename+"-"+str(load_epoch)+".pth")
vae.load_state_dict(vae_dict)
vae.to(device)

dalle = DALLE(
    dim = 256, #512,
    vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens = 10000,    # vocab size for text
    text_seq_len = 256,         # text sequence length
    depth = 6,                  # should be 64
    heads = 8,                  # attention heads
    dim_head = 64,              # attention head dimension
    attn_dropout = 0.1,         # attention dropout
    ff_dropout = 0.1            # feedforward dropout
)


# load pretrained dalle if continuing training

dalle_dict = torch.load(loadfn)
dalle.load_state_dict(dalle_dict)

dalle.to(device)

# get image and text data

lf = open("od-captionsonly.txt", "r")   # file contains captions only, one caption per line

# build vocabulary

from Vocabulary import Vocabulary

vocab = Vocabulary("captions")

captions = []
for lin in lf:
    captions.append(lin)
	
for caption in captions:
    vocab.add_sentence(caption)    
    
def tokenizer(text): # create a tokenizer function
    return text.split(' ')

inp_text = sys.argv[1]
print(inp_text)
tokens = tokenizer(inp_text)
codes = []
for t in tokens:
    codes.append(vocab.to_index(t))

print(codes)
c_tokens = [0]*256  # fill to match text_seq_len
c_tokens[:len(codes)] = codes    
       
text = torch.LongTensor(codes).unsqueeze(0).to(device)  # a minibatch of text (numerical tokens)
mask = torch.ones_like(text).bool().to(device)
oimgs = dalle.generate_images(text, mask = mask)
ts = int(time.time())
print(inp_text, ts)
save_image(oimgs,
               'results/gendalle'+name+'_epoch_' + str(dalle_epoch) + '-' +str(ts)+'.png', normalize=True)

