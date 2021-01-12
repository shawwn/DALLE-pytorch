import argparse
import torch
from dalle_pytorch import DiscreteVAE, DALLE
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='train VAE for DALLE-pytorch')
parser.add_argument('--batchSize', type=int, default=24, help='batch size for training (default: 24)')
parser.add_argument('--dataPath', type=str, default="./imagedata", help='path to imageFolder (default: ./imagedata')
parser.add_argument('--imageSize', type=int, default=256, help='image size for training (default: 256)')
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs (default: 500)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
#parser.add_argument('--tempsched', action='store_true', default=False, help='use temperature scheduling')
#parser.add_argument('--temperature', type=float, default=0.9, help='vae temperature (default: 0.9)')
parser.add_argument('--vaename', type=str, default="vae", help='experiment name')
parser.add_argument('--vae_epoch', type=int, default=0, help='start epoch numbering for continuing training (default: 0)')
parser.add_argument('--name', type=str, default="test", help='experiment name')
parser.add_argument('--load_dalle', type=str, default="", help='name for pretrained VAE when continuing training')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch numbering for continuing training (default: 0)')
opt = parser.parse_args()

# vae

load_epoch = opt.vae_epoch #499
vaename = opt.vaename #"v2vae256"

# general

imgSize = opt.imageSize #256
batchSize = opt.batchSize #24
n_epochs = opt.n_epochs #500
log_interval = 10
lr = opt.lr #1e-4

#dalle

# to continue training from a saved checkpoint, give checkpoint path as loadfn and start_epoch 

#loadfn = "./models/dalle_vae-cdim256-140.pth"   
#start_epoch = 140
loadfn = opt.load_dalle
start_epoch = opt.start_epoch
name = opt.name #v2vae256


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tf = transforms.Compose([
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
  ])

vae = DiscreteVAE(
    image_size = opt.imageSize,
    num_layers = 3,
    channels = 3,
    num_tokens = 2048,
    codebook_dim = 256,
    hidden_dim = 128,
    temperature = 0.9
)

# load pretrained vae
print("loading VAE from ./models/"+vaename+"-"+str(load_epoch)+".pth")
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
if loadfn != "":
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


lf = open("od-captions.txt", "r") # files contains lines in the format image_path : captions

data = []

for lin in lf:
    (fn, txt) = lin.split(":")
    tokens = tokenizer(txt)
    codes = []
    for t in tokens:
        #print(t)
        if t=="":
            continue
        codes.append(vocab.to_index(t))
    #print(fn, codes)
    data.append((fn, codes))



len_data = len(data)
print(len_data)
#datactr = 0

# an iterator for fetching data during training

class ImageCaptions:
    
    def __init__(self, data, batchsize=4):
        self.data = data
        self.len = len(data)
        self.index = 0
        self.end = False
        self.batchsize = batchsize
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.end:
            self.index = 0
            raise StopIteration
        i_data = []
        c_data = []
        for i in range(0, self.batchsize):
            i_data.append(self.data[self.index][0])
            c_tokens = [0]*256  # fill to match text_seq_len
            c_tokens_ = self.data[self.index][1]
            c_tokens[:len(c_tokens_)] = c_tokens_    
            c_data.append(c_tokens) 
            self.index += 1
            if self.index == self.len:
                self.end = True
                break     
        return i_data, c_data


optimizer = optim.Adam(dalle.parameters(), lr=lr)

for epoch in range(start_epoch, start_epoch+n_epochs):
  batch_idx = 0    
  train_loss = 0    
  dset = ImageCaptions(data, batchsize=batchSize) # initialize iterator
  
  for i,c in dset:  # loop through dataset by minibatch
    text = torch.LongTensor(c)  # a minibatch of text (numerical tokens)
    images = torch.zeros(len(i), 3, 256, 256) # placeholder for images
    
    text = text.to(device)
    #print(text)
    
    # fetch images into tensor based on paths given in minibatch
    ix = 0
    for imgfn in i:       # iterate through image paths in minibatch

        # note: images are expected to be in ./imagefolder/0/
        img_t = read_image(opt.dataPath+"/0/"+imgfn).float()/255.   # read image and scale into float 0..1
        img_t = tf(img_t)  # normalize 
        images[ix,:,:,:] = img_t 
        ix += 1

    images = images.to(device)
        
    mask = torch.ones_like(text).bool().to(device)
    
    # train and optimize a single minibatch
    optimizer.zero_grad()
    loss = dalle(text, images, mask = mask, return_loss = True)
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
    
    if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(i), len(data),
            100. * batch_idx / int(round(len(data)/batchSize)),
            loss.item() / len(i)))
    
    batch_idx += 1

  print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(data)))

  torch.save(dalle.state_dict(), "./models/"+name+"_dalle_"+str(epoch)+".pth")
  
  # generate a test sample from the captions in the last minibatch
  oimgs = dalle.generate_images(text, mask = mask)
  save_image(oimgs,
               'results/'+name+'_dalle_epoch_' + str(epoch) + '.png', normalize=True)

