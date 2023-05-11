# Description: Contains functions for training the denoiser model
# author: Kolade Gideon @Allaye
# github: www.github.com/allaye
# created: 2023-03-22
# last modified: 2023-03-26
import pickle
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataloader import DenoiserDataset
from denoiser import Denoiser
from utils import hyperparameter_tuning, experiment

torch.cuda.empty_cache()

num_epochs, learning_rate, batch_size, num_workers, device = hyperparameter_tuning()
dataset = DenoiserDataset(
    [r'../data/noisey_train/*.jpg', r'../data/train/*.jpg'],
    transform=transforms.ToTensor())
boardwriter = SummaryWriter(f'runs/denoiser')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
model = Denoiser()


def train(model, dataloader, num_epochs, lr, device='cpu', **kwargs):
    global loss
    # writer = SummaryWriter('runs/denoiser')
    criterion, optimizer = model.loss_optimizer(lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    step = 0
    for epoch in range(num_epochs):
        model = model.train()
        data_loop = tqdm(enumerate(dataloader))
        for idx, (noisy_image, clean_image) in data_loop:
            noisy_image = noisy_image
            clean_image = clean_image
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            reconstructed, mu, sigma = model(noisy_image)
            # toimg = transforms.ToPILImage()
            # img = toimg(noisy_image[0])
            # img1 = toimg(clean_image[0])
            # img.show()
            # img1.show()
            # compute loss
            print(f'clean_image: {clean_image.shape}')
            print(f'reconstructed: {reconstructed.shape}')
            reconstructed_loss = criterion(reconstructed, clean_image)
            kullback_leibler_divergence = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            # backward pass
            # loss.detach().item()
            print(f'kullback_leibler_divergence: {kullback_leibler_divergence:.4f}\n')
            print(f'reconstructed_loss: {reconstructed_loss:.4f}\n')
            loss = reconstructed_loss + kullback_leibler_divergence
            loss.backward()
            optimizer.step()
            data_loop.set_description(f'Epoch {epoch}, Step🐐 {idx} of {len(dataloader)}🎯')
            data_loop.set_postfix(loss=loss.item())
            boardwriter.add_scalar('training loss', loss.item(), global_step=step)
            boardwriter.add_scalar('reconstructed_loss', reconstructed_loss.item(), global_step=step)
            boardwriter.add_scalar('kullback_leibler_divergence', kullback_leibler_divergence.item(), global_step=step)
            step += 1

        print(f'epoch {epoch} done!, current loss of {loss.item():.4f}😒')
        # writer.add_scalar('training loss', loss.item(), epoch)
        # save model
        # torch.save(model.state_dict(), f'/models/denoiser_{epoch}.pth')
        with open(f'./models/denoiser_{epoch}.pths', 'wb') as handle:
            pickle.dump(model.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            kwargs['loss'] = loss.item()
            kwargs['epoch'] = epoch
            experiment(**kwargs)
    boardwriter.close()


if __name__ == '__main__':
    # print(model.__repr__())

    expe = {
        "architecture": model.__repr__(),
        "experiment": "exp_batchnorm_leaky_relu_pool_alpha",
        "hyperparameter": {
            "epoch": num_epochs,
            "lr": learning_rate,
             "batch_size": batch_size,
            "workers": num_workers,
            "device": str(device)
        }
    }
    # experiment(**expe)
    train(model, dataloader, num_epochs, learning_rate, **expe)

