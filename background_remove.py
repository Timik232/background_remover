import os
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import cv2
import numpy as np
import torch
from PIL import Image


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
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
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
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer module
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# Vision Transformer class
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 512, dim_head = 64, dropout = 0., emb_dropout = 0.):
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
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
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
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
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
        self.vit = ViT(image_size = 32,patch_size = 8,dim = 2048, depth = 2, heads = 16,mlp_dim = 12,channels = 512)
        self.vit_conv = nn.Conv2d(32,512,kernel_size = 1,padding = 0)
        self.vit_linear = nn.Linear(64,1024)

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

        #applying Vision Transformer
        x6 = self.vit(x5)
        x6 = torch.reshape(x6,(-1,32,8,8))
        x7 = self.vit_conv(x6)
        x8 = self.vit_linear(torch.reshape(x7,(-1,512,64)))
        x9 = torch.reshape(x8,(-1,512,32,32))

        # Decoder (Expanding Path)
        x = self.up1(x9, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# Функция для предобработки изображения перед отправкой в модель
def preprocess_image(image_path: str, device: torch.device, size=(512, 512)) -> torch.tensor:
    """
    Preprocess the image before sending it to the model
    :param image_path: path to image
    :param device: cpu or cuda
    :param size: resolution of the image
    :return: changed image
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]
    image = image.to(device)
    return image


# Функция для постобработки маски, полученной от модели
def postprocess_mask(mask: torch.tensor, original_size: tuple) -> np.ndarray:
    """
    make the binarization of mask and resize it to the original size of the source image
    :param mask: torch mask
    :param original_size: size of the original image
    :return:
    """
    mask = mask.squeeze().cpu().numpy()
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.uint8)  # Бинаризация маски
    return mask


# Функция для удаления фона с помощью маски
def remove_background(image_path: str, mask: torch.tensor) -> Image:
    """
    Remove the background from the image using the mask, change all not mask pixel to (0,0,0,0)
    :param image_path: path to image
    :param mask: torch tensor of mask
    :return: Image without background
    """
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    mask = np.array(mask).astype(np.uint8)  # Ensure mask is a NumPy array of type uint8
    mask = mask[:, :, 0] if mask.ndim == 3 else mask  # Extract the first channel if it's a 3D array
    for y in range(height):
        for x in range(width):
            if mask[y, x] != 0:
                image.putpixel((x, y), (0, 0, 0, 0))  # Set alpha channel to 0 for background
    return image


def get_image_without_background(image_path: str, model, device: torch.device, tempdir, count=0, user_id=0) -> str:
    """
    Get image without background
    :param image_path: path to image
    :param model: Unet model
    :param device: torch.device, cpu or cuda
    :param tempdir: temporary directory
    :param count: sequence number
    :param user_id: user id for creating directory
    :return: path to image without background
    """
    image_tensor = preprocess_image(image_path, device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        mask = torch.argmax(output, axis=1)[0]
    original_image = cv2.imread(image_path)
    original_size = (original_image.shape[1], original_image.shape[0])
    mask = postprocess_mask(mask, original_size)
    image_without_background = remove_background(image_path, mask)
    output_path = 'image_without_background.png'
    os.makedirs(os.path.join(tempdir.name, str(user_id), "without"), exist_ok=True)
    path = os.path.join(tempdir.name, str(user_id), "without", f"{count}_" + output_path)
    image_without_background.save(path)
    print(f"Изображение сохранено по пути: {path}")
    return path
