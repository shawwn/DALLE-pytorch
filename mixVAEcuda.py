import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dalle_pytorch import DiscreteVAE

imgSize = 256
load_epoch = 280

vae = DiscreteVAE(
    image_size = imgSize,
    num_layers = 3,
    channels = 3,
    num_tokens = 2048,
    codebook_dim = 1024,
    hidden_dim = 128
)

vae_dict = torch.load("./models/dvae-"+str(load_epoch)+".pth")
vae.load_state_dict(vae_dict)
vae.cuda()

batchSize = 12
n_epochs = 500
log_interval = 20
#images = torch.randn(4, 3, 256, 256)

t = transforms.Compose([
  transforms.Resize(imgSize),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #(0.267, 0.233, 0.234))
  ])

train_set = datasets.ImageFolder('./imagedata', transform=t, target_transform=None)

train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=batchSize, shuffle=True)



for batch_idx, (images, _) in enumerate(train_loader):
        images = images.cuda()
        codes = vae.get_codebook_indices(images)
        sample1 = vae.decode(codes)
        #save_image(sample.view(-1, 3, imgSize, imgSize),
        #       'results/recon_sample_' + str(batch_idx) + '.png', normalize=True)
        for i in range(0, 8):
            j = i + 1
            j = j % 8
            codes[i,512:] = codes[j,512:]
        sample2 = vae.decode(codes)
        grid = torch.cat([images[:8], sample1[:8], sample2[:8]])
        save_image(grid.view(-1, 3, imgSize, imgSize),
               'mixed/mixed_epoch_' +str(load_epoch) + "_"+ str(batch_idx) + '.png', normalize=True)
        #break


    
