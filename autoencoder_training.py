import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image

from autoencoder import ConvAutoencoder
from autoencoder_dataset import AutoencoderDataset

import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    """ transform = transforms.Compose([
        #transforms.Resize((480, 480)),
        transforms.ToTensor()
        #transforms.Normalize((0.5), (0.5))
    ]) """
    transform = transforms.ToTensor()

    #ae_data = AutoencoderDataset(img_dir='ae_images', transform=transform)
    ae_data = AutoencoderDataset(img_dir='ae_images')

    data_loader = DataLoader(dataset=ae_data, batch_size=16, shuffle=True)


    """ dataiter = iter(data_loader)
    img = dataiter.next()

    img = img[0].numpy().transpose(1, 2, 0)
    #img = img[0].permute(1, 2, 0).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img)
    cv2.imshow('Autoencoder input', img)
    cv2.waitKey(0) """
    

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    #criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    num_epochs = 100

    for epoch in range(num_epochs):
        for img in data_loader:
            #img = img.reshape(-1, 28*28).to(device)
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')


    torch.save(model.state_dict(), 'ae_model/ae_v1.pth')


    ae_data = AutoencoderDataset(img_dir='ae_images_test')
    data_loader = DataLoader(dataset=ae_data, batch_size=1, shuffle=True)
    dataiter = iter(data_loader)
    test_img = dataiter.next()

    res = model(test_img.to(device))

    res = res[0].cpu().detach().numpy().transpose(1, 2, 0)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    test_img = test_img[0].cpu().detach().numpy().transpose(1, 2, 0)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    cv2.imshow('Autoencoder | left: input, right: output', np.hstack((test_img, res)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()