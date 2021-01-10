import torch
from dalle_pytorch import DiscreteVAE, DALLE
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.utils import save_image

load_epoch = 390
vaename = "vae-cdim256"

imgSize = 256
batchSize = 12
n_epochs = 100
log_interval = 10
lr = 1e-4
name = "vae-cdim256"

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

dalle.to(device)

# get image and text data

lf = open("od-captionsonly.txt", "r")

from Vocabulary import Vocabulary

vocab = Vocabulary("captions")

captions = []
for lin in lf:
    captions.append(lin)
	
for caption in captions:
    vocab.add_sentence(caption)    
    
def tokenizer(text): # create a tokenizer function
    return text.split(' ')
    
lf = open("od-captions.txt", "r")

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
datactr = 0

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

for epoch in range(0, n_epochs):
  batch_idx = 0    
  train_loss = 0    
  dset = ImageCaptions(data, batchsize=batchSize)
  for i,c in dset:
    #print(i, c)
    text = torch.LongTensor(c)
    images = torch.zeros(len(i), 3, 256, 256)
    
    text = text.to(device)
    #print(text)
    
    ix = 0
    for imgfn in i:
        #print(imgfn)
        img_t = read_image("./imagedata/0/"+imgfn).float()/255.
        img_t = tf(img_t)
        #print(img_t.shape, img_t.mean())
        images[ix,:,:,:] = img_t
        ix += 1

    images = images.to(device)
        
    mask = torch.ones_like(text).bool().to(device)
    
    optimizer.zero_grad()
    loss = dalle(text, images, mask = mask, return_loss = True)
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
    
    if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(i), len(data),
            100. * batch_idx / len(data),
            loss.item() / len(i)))
    
    batch_idx += 1

  print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(data)))

  torch.save(dalle.state_dict(), "./models/dalle_"+name+"-"+str(epoch)+".pth")
  
  oimgs = dalle.generate_images(text, mask = mask)
  #print(epoch, ) # (2, 3, 256, 256)
  save_image(oimgs,
               'results/dalle'+name+'_epoch_' + str(epoch) + '.png', normalize=True)

