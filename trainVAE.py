import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dalle_pytorch import DiscreteVAE
from torch.nn.utils import clip_grad_norm_

parser = argparse.ArgumentParser(description='train VAE for DALLE-pytorch')
parser.add_argument('--batchSize', type=int, default=24, help='batch size for training (default: 24)')
parser.add_argument('--dataPath', type=str, default="./imagedata", help='path to imageFolder (default: ./imagedata')
parser.add_argument('--imageSize', type=int, default=256, help='image size for training (default: 256)')
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs (default: 500)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--tempsched', action='store_true', default=False, help='use temperature scheduling')
parser.add_argument('--temperature', type=float, default=0.9, help='vae temperature (default: 0.9)')
parser.add_argument('--name', type=str, default="vae", help='experiment name')
parser.add_argument('--loadVAE', type=str, default="", help='name for pretrained VAE when continuing training')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch numbering for continuing training (default: 0)')
parser.add_argument('--clip', type=float, default=0, help='clip weights, 0 = no clipping (default: 0)')
opt = parser.parse_args()

imgSize = opt.imageSize #256
batchSize = opt.batchSize #24
n_epochs = opt.n_epochs #500
log_interval = 10
lr = opt.lr #1e-4
temperature_scheduling = opt.tempsched #True

name = opt.name #"v2vae256"

# for continuing training 
# set loadfn: path to pretrained model
# start_epoch: start epoch numbering from this
loadfn = opt.loadVAE #""
start_epoch = opt.start_epoch #0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = DiscreteVAE(
    image_size = imgSize,
    num_layers = 3,
    channels = 3,
    num_tokens = 2048,
    codebook_dim = 256,
    hidden_dim = 128,
    temperature = opt.temperature
)

if loadfn != "":
    vae_dict = torch.load(loadfn)
    vae.load_state_dict(vae_dict)


vae.to(device)


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
    return transforms.functional.center_crop(img, min(img.size))

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
    size = (min(img.size), min(img.size))
    # Only step forward along this edge if it's the long edge
    i = (0 if size[0] == img.size[0]
          else np.random.randint(low=0,high=img.size[0] - size[0]))
    j = (0 if size[1] == img.size[1]
          else np.random.randint(low=0,high=img.size[1] - size[1]))
    return transforms.functional.crop(img, i, j, size[0], size[1])

  def __repr__(self):
    return self.__class__.__name__


t = transforms.Compose([
  CenterCropLongEdge(),
  transforms.Resize(imgSize),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #(0.267, 0.233, 0.234)
  ])

train_set = datasets.ImageFolder(opt.dataPath, transform=t, target_transform=None)

train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=batchSize, shuffle=True)

optimizer = optim.Adam(vae.parameters(), lr=lr)

def clampWeights(m):
    if type(m) != nn.BatchNorm2d and type(m) != nn.Sequential:
      for p in m.parameters():
        p.data.clamp_(-opt.clip, opt.clip)

if temperature_scheduling:
    vae.temperature = opt.temperature
    dk = 0.7 ** (1/len(train_loader)) 
    print('Scale Factor:', dk)

for epoch in range(start_epoch, start_epoch + n_epochs):

    train_loss = 0
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device) 
        recons = vae(images)
        loss = F.smooth_l1_loss(images, recons) + F.mse_loss(images, recons)
        

        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if opt.clip > 0:
            vae.apply(clampWeights)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(images)))

    if temperature_scheduling: 
            vae.temperature *= dk
            print("Current temperature: ", vae.temperature)

    k = 8
    with torch.no_grad():
        codes = vae.get_codebook_indices(images)
        imgx = vae.decode(codes)
    grid = torch.cat([images[:k], recons[:k], imgx[:k]])
    save_image(grid,
               'results/'+name+'_epoch_' + str(epoch) + '.png', normalize=True)

    print('====> Epoch: {} Average loss: {:.8f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    torch.save(vae.state_dict(), "./models/"+name+"-"+str(epoch)+".pth")



    
