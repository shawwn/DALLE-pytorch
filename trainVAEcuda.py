import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dalle_pytorch import DiscreteVAE

imgSize = 256

vae = DiscreteVAE(
    image_size = imgSize,
    num_layers = 3,
    num_tokens = 1024,
    codebook_dim = 512,
    hidden_dim = 64
)

vae.cuda()

batchSize = 32
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

optimizer = optim.Adam(vae.parameters(), lr=1e-5)
#optimizer.cuda()

# train with a lot of data to learn a good codebook
for epoch in range(0, n_epochs):

    train_loss = 0
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.cuda()
        loss = vae(images, return_recon_loss = True)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(images)))

    sample = vae(images, return_recon_loss = False)
    sample = sample / 2 + 0.5
    save_image(sample.view(-1, 3, imgSize, imgSize),
               'results/sample_' + str(epoch) + '.png')
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    torch.save(vae.state_dict(), "./models/dvae-"+str(epoch)+".pth")


    
