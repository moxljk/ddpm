import torch
import torchvision
from tqdm import tqdm
from unet import Unet

MAX_T = 1000
BETA_START = 1e-4
BETA_END = 0.02
DEVICE = 'cuda'
torch.set_default_device(DEVICE)

def alpha_bar(timesteps):
    return torch.prod(1 - torch.linspace(BETA_START, BETA_END, MAX_T)[:timesteps])

dataset = torchvision.datasets.ImageFolder("data/celeba_hq_256", torchvision.transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

model = Unet(channels_level=[64, 128, 256, 512, 1024, 1024])
model.to(DEVICE)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr = 2e-5)
loss = torch.nn.MSELoss()

for epoch in tqdm(range(1000)):
    tloader = tqdm(loader, desc=f"Epoch {epoch+1}")
    tloader.set_postfix_str(f"MSELoss: -")

    for image, _ in tloader:
        B, C, H, W = image.shape
        ts = torch.randint(0, MAX_T, (B,))
        eps = torch.randn(image.shape)
        alpha_bars = torch.tensor([alpha_bar(i) for i in ts])[:,None,None,None]
        xs = torch.sqrt(alpha_bars) * image.to(DEVICE) + torch.sqrt(1 - alpha_bars) * eps

        l = loss(eps, model(xs, ts))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        tloader.set_postfix_str(f"MSELoss: {l}")
    torch.save(model, f"./weights/epoch_{epoch+1}.pth")