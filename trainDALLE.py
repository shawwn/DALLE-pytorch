import argparse
import torch
from dalle_pytorch import DiscreteVAE, DALLE
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.utils import save_image
import os

mode = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = mode == 'cuda'
stub = False

parser = argparse.ArgumentParser(description='train VAE for DALLE-pytorch')
parser.add_argument('--batchSize', type=int, default=24 if cuda else 1, help='batch size for training (default: 24)')
parser.add_argument('--text_seq_len', type=int, default=256 if not stub else 50, help='text sequence length (default: 256)')
parser.add_argument('--dataPath', type=str, default="./imagedata", help='path to imageFolder (default: ./imagedata)')
parser.add_argument('--captionPath', type=str, default="captions.txt", help='path to captions file containing lines like "<path to image>:<caption for that image>" (default: captions.txt)')
parser.add_argument('--imageSize', type=int, default=256, help='image size for training (default: 256)')
parser.add_argument('--n_epochs', type=int, default=100_000_000, help='number of epochs (default: 100,000,000)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
#parser.add_argument('--tempsched', action='store_true', default=False, help='use temperature scheduling')
#parser.add_argument('--temperature', type=float, default=0.9, help='vae temperature (default: 0.9)')
parser.add_argument('--vaename', type=str, default="vae", help='experiment name')
parser.add_argument('--vae_epoch', type=int, default=0, help='start epoch numbering for continuing training (default: 0)')
parser.add_argument('--name', type=str, default="test", help='experiment name')
parser.add_argument('--load_dalle', type=str, default="", help='name for pretrained VAE when continuing training')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch numbering for continuing training (default: 0)')
parser.add_argument('--vocabSize', type=int, default=49408, help='vocab size (default: CLIP)')
parser.add_argument('--save_every_n_epochs', type=int, default=1, help='save model every N epochs (default: 1)')
parser.add_argument('--gen_every_n_epochs', type=int, default=1, help='generate samples every N epochs (default: 1)')
parser.add_argument('--generate', action='store_true', help='generate an image interactively')
opt = parser.parse_args()

from pprint import pprint as pp

import inspect
import traceback

def p(x):
  if inspect.isgenerator(x):
    x = list(x)
  pp(x)
  return x

import time

def now():
  return time.time()

def log(*args):
  with tqdm.tqdm.external_write_mode():
    print(*args)

def log_exn(*args):
  with tqdm.tqdm.external_write_mode():
    traceback.print_exc()
    if len(args) > 0:
      print(*args)

# vae

load_epoch = opt.vae_epoch #499
vaename = opt.vaename #"v2vae256"

# general

imgSize = opt.imageSize #256
batchSize = opt.batchSize #24
n_epochs = opt.n_epochs #500
log_interval = 10
lr = opt.lr #1e-4

# get image and text data

import encoder
tokenizer = encoder.get_encoder()

with open(opt.captionPath, "r") as f: # files contains lines in the format image_path : captions
  lf = [line for line in f if not line.startswith('#')]

data = []

import tqdm

def tokenize(text):
  return tokenizer.encode(text + '<|endoftext|>')

for lin in tqdm.tqdm(lf):
    lin = lin.rstrip('\r\n')
    (fn, txt) = lin.split(":", 1) if ':' in lin else (lin, lin)
    txt = txt or fn
    codes = tokenize(txt)
    if len(codes) > opt.text_seq_len:
      eot = codes.pop() # get endoftext token
      codes = codes[0:opt.text_seq_len-1] # truncate 
      codes[-1] = eot # set last token to endoftext
    #print(fn, codes)
    data.append((fn, codes))

print(len(data))
#datactr = 0

# an iterator for fetching data during training

class ImageCaptions:
    
    def __init__(self, data, batchsize=4):
        self.data = data
        self.len = len(data)
        self.index = 0
        self.end = False
        self.batchsize = batchsize
        
    def __len__(self):
        return self.len

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
            c_tokens = [0]*opt.text_seq_len  # fill to match text_seq_len
            c_tokens_ = self.data[self.index][1]
            c_tokens[:len(c_tokens_)] = c_tokens_    
            c_data.append(c_tokens) 
            self.index += 1
            if self.index == self.len:
                self.end = True
                break     
        return i_data, c_data

#dalle

# to continue training from a saved checkpoint, give checkpoint path as loadfn and start_epoch 

#loadfn = "./models/dalle_vae-cdim256-140.pth"   
#start_epoch = 140
loadfn = opt.load_dalle
start_epoch = opt.start_epoch
name = opt.name #v2vae256


device = torch.device(mode)

class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    size = img.size() if callable(img.size) else img.size
    if len(size) > 2:
      C, H, W = size
      assert C in [1, 3, 4]
      size = (H, W)
    return transforms.functional.center_crop(img, min(size))

  def __repr__(self):
    return self.__class__.__name__



class RandomCropLongEdge(object):
  """Crops the given PIL Image on the long edge with a random start point.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    size = img.size() if callable(img.size) else img.size
    size = (min(size), min(size))
    # Only step forward along this edge if it's the long edge
    i = (0 if size[0] == img.size[0]
          else np.random.randint(low=0,high=img.size[0] - size[0]))
    j = (0 if size[1] == img.size[1]
          else np.random.randint(low=0,high=img.size[1] - size[1]))
    return transforms.functional.crop(img, i, j, size[0], size[1])

  def __repr__(self):
    return self.__class__.__name__


tf = transforms.Compose([
  CenterCropLongEdge(),
  transforms.Resize(imgSize),
  #transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #(0.267, 0.233, 0.234)
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
vae_dict = torch.load("./models/"+vaename+"-"+str(load_epoch)+".pth", map_location=device if mode == 'cpu' else None)
vae.load_state_dict(vae_dict)
vae.to(device)

print('Creating DALLE...')

if cuda or opt.generate:
  dalle = DALLE(
      dim = 256, #512,
      vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
      num_text_tokens = opt.vocabSize,    # vocab size for text
      text_seq_len = opt.text_seq_len,         # text sequence length
      depth = 6,                  # should be 64
      heads = 8,                  # attention heads
      dim_head = 64,              # attention head dimension
      attn_dropout = 0.1,         # attention dropout
      ff_dropout = 0.1            # feedforward dropout
  )
else:
  dalle = DALLE(
      dim = 256, #512,
      vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
      num_text_tokens = opt.vocabSize,    # vocab size for text
      text_seq_len = opt.text_seq_len,          # text sequence length
      depth = 2,                  # should be 64
      heads = 2,                  # attention heads
      dim_head = 2 ,              # attention head dimension
      attn_dropout = 0.1,         # attention dropout
      ff_dropout = 0.1            # feedforward dropout
  )


# load pretrained dalle if continuing training
if loadfn != "":
    print('Loading DALLE...')
    dalle_dict = torch.load(loadfn, map_location=device if mode == 'cpu' else None)
    dalle.load_state_dict(dalle_dict)

dalle.to(device)


if opt.generate:
  while True:
    caption = input('Type a prompt:').rstrip()
    tokens = tokenize(caption)
    while len(tokens) < opt.text_seq_len:
      tokens += [0]
    print(tokens)
    print(repr(tokenizer.decode(tokens, sep='')))
    texts = torch.LongTensor([tokens]).to(device)  # a minibatch of text (numerical tokens)
    mask = torch.ones_like(texts).bool().to(device)
    fpath = 'results/prompt.png'
    log('Generating {!r}...'.format(fpath))
    oimgs = dalle.generate_images(texts, mask = mask)
    log('Saving {!r}'.format(fpath))
    save_image(oimgs, fpath, normalize=True)

optimizer = optim.Adam(dalle.parameters(), lr=lr)

last_print = now()

v_loss = float('inf')

ebar = tqdm.trange(start_epoch, start_epoch+n_epochs, position=0, desc='Epoch')
for epoch in ebar:
  batch_idx = 0    
  train_loss = 0    
  dset = ImageCaptions(data, batchsize=batchSize) # initialize iterator
  with tqdm.tqdm(total=len(dset), position=1, desc='Batch') as pbar:
    
    for i,c in dset:  # loop through dataset by minibatch
      texts = torch.LongTensor(c)  # a minibatch of text (numerical tokens)
      images = torch.zeros(len(i), 3, imgSize, imgSize) # placeholder for images
      
      texts = texts.to(device)
      
      # fetch images into tensor based on paths given in minibatch
      for ix, imgfn in enumerate(i):       # iterate through image paths in minibatch
        if ix == 0:
          caption = tokenizer.decode(list(texts[ix].cpu().numpy()), sep='')
          log('loss: {:.6f}'.format(v_loss), imgfn, caption)
        img_t = read_image(os.path.join(opt.dataPath,imgfn)).float() / 255.0
        try:
          img_t = tf(img_t)  # normalize 
          images[ix,:,:,:] = img_t 
        except:
          log_exn('Image failed: ', imgfn)
      
      if now() - last_print > 1.0 and mode == 'cpu':
        with torch.no_grad():
            recons = vae(images)
            codes = vae.get_codebook_indices(images)
            imgx = vae.decode(codes)
        k = 8
        grid = torch.cat([images[:k], recons[:k], imgx[:k]])
        save_image(grid, 'reals.png')
        last_print = now()
        

      images = images.to(device)
          
      mask = torch.ones_like(texts).bool().to(device)

      if not stub:
        # train and optimize a single minibatch
        optimizer.zero_grad()
        loss = dalle(texts, images, mask = mask, return_loss = True)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        v_loss = loss.item() / len(i)
      else:
        v_loss = float('inf')
      
      msg = ''
      if False:
        msg += ('Train Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
            epoch, batch_idx * len(i), len(data),
            100. * batch_idx / int(round(len(data)/batchSize))))
      msg += ('Batch loss: {:.6f}'.format(v_loss))
      pbar.set_description(msg)
      pbar.update(dset.batchsize)
      pbar.refresh()
      ebar.refresh()

      if batch_idx % log_interval == 0:
        log(msg)
      
      batch_idx += 1

  msg = '====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data))
  log(msg)
  ebar.set_description(msg)
  ebar.refresh()

  if not stub:
    if epoch % opt.save_every_n_epochs == 0:
      fpath = "./models/"+name+"_dalle_"+str(epoch)+".pth"
      log('Saving {!r}'.format(fpath))
      torch.save(dalle.state_dict(), fpath)
    
    # generate a test sample from the captions in the last minibatch
    if epoch % opt.gen_every_n_epochs == 0:
      fpath = 'results/'+name+'_dalle_epoch_' + str(epoch) + '.png'
      log('Generating {!r}...'.format(fpath))
      oimgs = dalle.generate_images(texts, mask = mask)
      log('Saving {!r}'.format(fpath))
      save_image(oimgs, fpath, normalize=True)

