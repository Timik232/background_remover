import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from random import shuffle
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import math
from glob import glob
import sys
import shutil
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import PIL
import random
from scipy import ndimage
from sending_functions import send_message

def train(filename: str, user_id: int, temp_dir, N_EPOCHS=10, BATCH_SIZE=4):
    class segDataset(torch.utils.data.Dataset):
        def __init__(self, root, training, transform=None):
            super(segDataset, self).__init__()
            self.root = root
            self.training = training
            self.transform = transform
            self.IMG_NAMES = sorted(glob(self.root + '/*/images/*.jpg'))

            # Define the BGR color values for different classes in the dataset
            self.BGR_classes = {'Ladder': [255, 0, 43],
                                'Background': [0, 0, 0]}  # in BGR

            # Define the binary classes for segmentation
            self.bin_classes = ['Ladder', 'Background']

        def __getitem__(self, idx):
            img_path = self.IMG_NAMES[idx]
            mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')

            # Read the image and its mask using OpenCV
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            # print(mask_path)

            # Create a mask for each class based on the color values in the image
            cls_mask = np.zeros(mask.shape)
            cls_mask[mask == self.BGR_classes['Ladder']] = self.bin_classes.index('Ladder')
            cls_mask[mask == self.BGR_classes['Background']] = self.bin_classes.index('Background')
            cls_mask = cls_mask[:, :, 0]

            # Apply data augmentation if it's a training dataset
            if self.training == True:
                if self.transform:
                    # Apply transformations to the image
                    image = transforms.functional.to_pil_image(image)
                    image = self.transform(image)
                    image = np.array(image)

                # Apply random rotations
                if np.random.rand() < 0.5:
                    angle = np.random.randint(4) * 90
                    image = ndimage.rotate(image, angle, reshape=True)
                    cls_mask = ndimage.rotate(cls_mask, angle, reshape=True)

                # Apply vertical flips
                if np.random.rand() < 0.5:
                    image = np.flip(image, 0)
                    cls_mask = np.flip(cls_mask, 0)

                # Apply horizontal flips
                if np.random.rand() < 0.5:
                    image = np.flip(image, 1)
                    cls_mask = np.flip(cls_mask, 1)

            # Resize the image and mask
            image = cv2.resize(image, (512, 512)) / 255.0
            cls_mask = cv2.resize(cls_mask, (512, 512))
            image = np.moveaxis(image, -1, 0)

            # Return the image and its corresponding mask as torch tensors
            return torch.tensor(image).float(), torch.tensor(cls_mask, dtype=torch.int64)

        def __len__(self):
            # Return the total number of images in the dataset
            return len(self.IMG_NAMES)

    # Define color shift transformation
    color_shift = transforms.ColorJitter(.1, .1, .1, .1)

    # Define blurriness transformation
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

    # Compose the transformations
    t = transforms.Compose([color_shift, blurriness])
    dataset = segDataset(filename, training=True, transform=t)

    # Get the length of the dataset
    print(f"Length of the dataset: {len(dataset)}")
    # Calculate the number of samples for the test set (10% of the total dataset)
    test_num = int(0.1 * len(dataset))

    # Print the number of samples in the test set
    print(f'test data : {test_num}')

    # The first part is for training and the second part is for testing
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_num, test_num],
                                                                generator=torch.Generator().manual_seed(101))

    # Create a DataLoader for the training dataset
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Create a DataLoader for the test dataset
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Function to get the default device available (GPU if available, else CPU)
    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    # Function to move data to the specified device
    def to_device(data, device):
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    # Wrapper class for DataLoader to move batches of data to the specified device
    class DeviceDataLoader():
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            return len(self.dl)

    # Get the default device
    device = get_default_device()

    # Move the training and test DataLoader batches to the selected device
    train_dataloader = DeviceDataLoader(train_dataloader, device)
    test_dataloader = DeviceDataLoader(test_dataloader, device)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the DoubleConv block, which consists of two convolutional layers followed by batch normalization and ReLU activation
    class DoubleConv(nn.Module):
        """
        (convolution => [BN] => ReLU) * 2
        """

        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)

    # Define the Down block, which downscales the input using max pooling followed by a DoubleConv block
    class Down(nn.Module):
        """
        Downscaling with maxpool then double conv
        """

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

        def forward(self, x):
            return self.maxpool_conv(x)

    # Define the Up block, which upscales the input and concatenates it with the skip connection from the encoder, followed by a DoubleConv block
    class Up(nn.Module):
        """
        Upscaling then double conv
        """

        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()

            # If bilinear interpolation is used, upscale using bilinear interpolation and reduce the number of channels
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                # Otherwise, use transposed convolution for upscaling
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            # Calculate the difference in size between the input and the skip connection
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            # Pad the input to match the size of the skip connection
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # Concatenate the skip connection with the upscaled input
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    # Define the final output convolution layer
    class OutConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(OutConv, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange

    # Function to ensure the input is a tuple
    def pair(t):
        return t if isinstance(t, tuple) else (t, t)

    # Pre-Normalization module
    class PreNorm(nn.Module):
        def __init__(self, dim, fn):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fn = fn

        def forward(self, x, **kwargs):
            return self.fn(self.norm(x), **kwargs)

    # Feed Forward Neural Network module
    class FeedForward(nn.Module):
        def __init__(self, dim, hidden_dim, dropout=0.):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            return self.net(x)

    # Attention mechanism module
    class Attention(nn.Module):
        def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
            super().__init__()
            inner_dim = dim_head * heads
            project_out = not (heads == 1 and dim_head == dim)

            self.heads = heads
            self.scale = dim_head ** -0.5

            self.attend = nn.Softmax(dim=-1)
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

        def forward(self, x):
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)

            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)

    # Transformer module
    class Transformer(nn.Module):
        def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
            super().__init__()
            self.layers = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

        def forward(self, x):
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return x

    # Vision Transformer class
    class ViT(nn.Module):
        def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=512, dim_head=64,
                     dropout=0., emb_dropout=0.):
            super().__init__()
            image_height, image_width = pair(image_size)
            patch_height, patch_width = pair(patch_size)

            # Check if image dimensions are divisible by patch dimensions
            assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

            # Calculate number of patches and patch dimension
            num_patches = (image_height // patch_height) * (image_width // patch_width)
            patch_dim = channels * patch_height * patch_width
            assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

            # Patch embedding layer
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.Linear(patch_dim, dim),
            )

            # Positional embedding
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.dropout = nn.Dropout(emb_dropout)

            # Transformer
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

            self.pool = pool
            self.to_latent = nn.Identity()

        def forward(self, img):
            x = self.to_patch_embedding(img)
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            return x

    class UNet(nn.Module):
        def __init__(self, n_channels, n_classes, bilinear=True):
            super(UNet, self).__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.bilinear = bilinear

            # Encoder (Contracting Path)
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)

            # Vision Transformer block
            self.vit = ViT(image_size=32, patch_size=8, dim=2048, depth=2, heads=16, mlp_dim=12, channels=512)
            self.vit_conv = nn.Conv2d(32, 512, kernel_size=1, padding=0)
            self.vit_linear = nn.Linear(64, 1024)

            # Decoder (Expanding Path)
            self.up1 = Up(1024, 512 // factor, bilinear)
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            self.up4 = Up(128, 64, bilinear)
            self.outc = OutConv(64, n_classes)

        def forward(self, x):
            # Encoder (Contracting Path)
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            # applying Vision Transformer
            x6 = self.vit(x5)
            x6 = torch.reshape(x6, (-1, 32, 8, 8))
            x7 = self.vit_conv(x6)
            x8 = self.vit_linear(torch.reshape(x7, (-1, 512, 64)))
            x9 = torch.reshape(x8, (-1, 512, 32, 32))

            # Decoder (Expanding Path)
            x = self.up1(x9, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits

    class FocalLoss(nn.Module):
        def __init__(self, gamma=0, alpha=None, size_average=True):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha

            # If alpha is a single value (float or int), convert it to a tensor
            if isinstance(alpha, (float, int)):
                self.alpha = torch.Tensor([alpha, 1 - alpha])

            # If alpha is a list, convert it to a tensor
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            self.size_average = size_average

        def forward(self, input, target):
            if input.dim() > 2:
                # If input has more than 2 dimensions (e.g., image), reshape it for calculation
                input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
                input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
                input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
            target = target.view(-1, 1)

            logpt = F.log_softmax(input, dim=-1)
            logpt = logpt.gather(1, target)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())

            if self.alpha is not None:
                # If alpha is specified, weight the loss for each class
                if self.alpha.type() != input.data.type():
                    self.alpha = self.alpha.type_as(input.data)
                at = self.alpha.gather(0, target.data.view(-1))
                logpt = logpt * Variable(at)

            # Compute the focal loss
            loss = -1 * (1 - pt) ** self.gamma * logpt

            # Calculate the mean loss if size_average is True, else sum the loss
            if self.size_average:
                return loss.mean()
            else:
                return loss.sum()

    criterion = FocalLoss(gamma=3 / 4).to(device)

    def acc(label, predicted):
        seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(y.cpu())
        return seg_acc

    # Define precision function
    def precision(y, pred_mask, classes=6):
        precision_list = [];
        for i in range(classes):
            # Compute precision for each class
            actual_num = y.cpu() == i
            predicted_num = i == torch.argmax(pred_mask, axis=1).cpu()

            # Calculate precision for class i
            prec = torch.logical_and(actual_num, predicted_num).sum() / predicted_num.sum()
            precision_list.append(prec.numpy().tolist())
        return precision_list

    # Define recall function
    def recall(y, pred_mask, classes=6):
        recall_list = []
        for i in range(classes):
            # Compute recall for each class
            actual_num = y.cpu() == i
            predicted_num = i == torch.argmax(pred_mask, axis=1).cpu()

            # Calculate recall for class i
            recall_val = torch.logical_and(actual_num, predicted_num).sum() / actual_num.sum()
            recall_list.append(recall_val.numpy().tolist())
        return recall_list

    # Initialize minimum loss as infinity
    min_loss = torch.tensor(float('inf'))
    # Initialize UNet model with specified number of input channels, output classes, and bilinear interpolation flag
    model = UNet(n_channels=3, n_classes=2, bilinear=True).to(device)
    # Initialize Adam optimizer for training the model with a learning rate of 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Initialize learning rate scheduler to adjust learning rate during training
    # StepLR reduces the learning rate by a factor of 0.5 every epoch
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    os.makedirs(f'{user_id}/saved_models', exist_ok=True)


    # Number of samples in the training and testing datasets
    N_DATA = len(train_dataset)
    N_TEST = len(test_dataset)

    # List to store losses for plotting
    plot_losses = []

    # Counter for the scheduler
    scheduler_counter = 0
    best_name = ""
    for epoch in range(N_EPOCHS):
        # Set model to training mode
        model.train()

        # Lists to store losses and accuracies for each batch
        loss_list = []
        acc_list = []

        # Iterate over batches in the training dataset
        for batch_i, (x, y) in enumerate(train_dataloader):
            # Forward pass: compute predicted outputs by passing inputs to the model
            pred_mask = model(x.to(device))

            # Calculate the loss
            loss = criterion(pred_mask, y.to(device))

            # Zero the gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Append the loss and accuracy to the respective lists
            loss_list.append(loss.cpu().detach().numpy())
            acc_list.append(acc(y, pred_mask).numpy())

            # Print progress
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    (epoch + 1),
                    N_EPOCHS,
                    batch_i,
                    len(train_dataloader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                )
            )

        # Increment scheduler counter
        scheduler_counter += 1

        # Set model to evaluation mode
        model.eval()

        # Lists to store validation losses and accuracies for each batch
        val_loss_list = []
        val_acc_list = []

        # Iterate over batches in the test dataset
        for batch_i, (x, y) in enumerate(test_dataloader):
            with torch.no_grad():
                # Forward pass: compute predicted outputs by passing inputs to the model
                pred_mask = model(x.to(device))

            # Calculate the validation loss
            val_loss = criterion(pred_mask, y.to(device))

            # Append the validation loss and accuracy to the respective lists
            val_loss_list.append(val_loss.cpu().detach().numpy())
            val_acc_list.append(acc(y, pred_mask).numpy())

        # Print epoch statistics
        print(' epoch {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val acc : {:.2f}'.format((epoch + 1),
                                                                                                       np.mean(
                                                                                                           loss_list),
                                                                                                       np.mean(
                                                                                                           acc_list),
                                                                                                       np.mean(
                                                                                                           val_loss_list),
                                                                                                       np.mean(
                                                                                                           val_acc_list)))
        # Append epoch losses for plotting
        plot_losses.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])

        # Check if the current validation loss is the lowest encountered so far
        compare_loss = np.mean(val_loss_list)
        is_best = compare_loss < min_loss

        # If the current loss is the lowest, save the model
        if is_best == True:
            scheduler_counter = 0
            min_loss = min(compare_loss, min_loss)
            torch.save(model.state_dict(),
                       f'{user_id}' + '/saved_models/unet_epoch_{}_{:.5f}.pt'.format(epoch, np.mean(val_loss_list)))
            best_name = f'{user_id}' + '/saved_models/unet_epoch_{}_{:.5f}.pt'.format(epoch, np.mean(val_loss_list))
        # If the scheduler counter exceeds a certain threshold, adjust the learning rate
        if scheduler_counter > 5:
            lr_scheduler.step()
            print(f"lowering learning rate to {optimizer.param_groups[0]['lr']}")
            scheduler_counter = 0
    if best_name == "":
        torch.save(model.state_dict(),
                   f'{user_id}' + '/saved_models/unet.pt')
        best_name = f'{user_id}' + '/saved_models/unet.pt'
    source_folder = os.path.join(f'{user_id}', "saved_models")
    destination_folder = os.path.join("models", str(user_id))
    destination_file = os.path.join(destination_folder, best_name.split("/")[-1])
    shutil.copy2(best_name, destination_file)
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    send_message(user_id, "Обучение завершено. Напишите 'модели' для выбора модели.")
    temp_dir.cleanup()
