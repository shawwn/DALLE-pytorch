import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dalle_pytorch import DiscreteVAE
from torch.nn.utils import clip_grad_norm_

imgSize = 256
batchSize = 24
n_epochs = 500
log_interval = 10
lr = 1e-4
temperature_scheduling = True
name = "v2vae256"
loadfn = ""
load_epoch = 0




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = DiscreteVAE(
    image_size = imgSize,
    num_layers = 3,
    channels = 3,
    num_tokens = 2048,
    codebook_dim = 1024,
    hidden_dim = 128
)

if loadfn != "":
    vae_dict = torch.load(loadfn)
    vae.load_state_dict(vae_dict)


vae.to(device)

t = transforms.Compose([
  transforms.Resize(imgSize),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #(0.267, 0.233, 0.234))
  ])

train_set = datasets.ImageFolder('./imagedata', transform=t, target_transform=None)

train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=batchSize, shuffle=True)

optimizer = optim.Adam(vae.parameters(), lr=lr)

if temperature_scheduling:
    vae.temperature = 5
    dk = 0.7 ** (1/len(train_loader)) 
    print('Scale Factor:', dk)

for epoch in range(0, n_epochs):

    train_loss = 0
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device) 
        recons = vae(images)
        loss = F.smooth_l1_loss(images, recons) + F.mse_loss(images, recons)
        

        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    torch.save(vae.state_dict(), "./models/"+name+"-"+str(epoch)+".pth")



    
