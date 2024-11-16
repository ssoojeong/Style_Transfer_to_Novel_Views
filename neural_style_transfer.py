import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 로더
def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
    ])
    image = Image.open(image_name).resize((imsize, imsize))
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# 이미지를 PIL 형식으로 변환
unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.001)

# Content Loss 클래스
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# Gram Matrix 계산
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

# Style Loss 클래스
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Normalization 클래스
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

# 스타일 모델과 손실 생성
def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img):
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name == 'conv_4':
            target = model(content_img).detach()
            content_loss = ContentLoss(target).to(device)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature).to(device)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses

# 입력 이미지 최적화 설정
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_(True)])
    return optimizer

# 스타일 트랜스퍼 실행
def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps=300, style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

# 메인 함수
def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("--content_image_path", type=str, default='./samples/psc/set4/', help="Path to the content image")
    parser.add_argument("--style_image_path", type=str, default='./data/samples/wikiart/style.jpg', help="Path to the style image")
    parser.add_argument("--save_path", type=str, default='./samples/stylized', help="Path to save the stylized image")
    parser.add_argument("--imsize", type=int, default=512, help="Image size (default: 512)")
    parser.add_argument("--num_steps", type=int, default=300, help="Number of optimization steps")
    parser.add_argument("--style_weight", type=float, default=1e6, help="Weight for style loss")
    parser.add_argument("--content_weight", type=float, default=1, help="Weight for content loss")

    args = parser.parse_args()
     
    for img in ['warp_image12.png', 'warp_image21.png']:
        input = os.path.join(args.content_image_path, img)
        content_img = image_loader(input, args.imsize)
        style_img = image_loader(args.style_image_path, args.imsize)
        input_img = content_img.clone()

        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img, args.num_steps, args.style_weight, args.content_weight)

        # 스타일화된 이미지 저장
        output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
        save_path = os.path.join(args.save_path, input.split('/')[-2])
        os.makedirs(save_path, exist_ok=True)
        output_image.save(os.path.join(save_path, img))
        print(f"Stylized image saved at {args.save_path}")

if __name__ == "__main__":
    main()
