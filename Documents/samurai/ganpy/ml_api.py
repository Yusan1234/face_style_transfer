from flask import Flask, jsonify, request

import json
import base64
from io import BytesIO

 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
from PIL import Image
import matplotlib.pyplot as plt
 
import torchvision.transforms as transforms
import torchvision.models as models

import copy

# initialize our Flask application
app = Flask(__name__)


def preprocess(con_img, sty_img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imsize = 512 if torch.cuda.is_available() else 256  # use small size if no gpu
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    def image_loader(image_name):
        # image = Image.open(image_name)
        image = Image.open(BytesIO(base64.b64decode(image_name)))
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    style_img = image_loader(sty_img)
    content_img = image_loader(con_img)
    input_img = content_img.clone()
    img = Image.open(BytesIO(base64.b64decode(con_img))).convert('RGB')

    class ContentLoss(nn.Module):

        def __init__(self, target, ):
            super(ContentLoss, self).__init__()
            # we 'detach' the target content from the tree used
            # to dynamically compute the gradient: this is a stated value,
            # not a variable. Otherwise the forward method of the criterion
            # will throw an error.
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    def gram_matrix(input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    class StyleLoss(nn.Module):

        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # create a module to normalize input image so we can easily put it in a
    # nn.Sequential
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

        # desired depth layers to compute style/content losses :

    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)

        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=500,
                           style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        # print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                         normalization_mean, normalization_std,
                                                                         style_img,
                                                                         content_img)
        optimizer = get_input_optimizer(input_img)

        # print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                # if run[0] % 50 == 0:
                #     print("run {}:".format(run))
                #     print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                #         style_score.item(), content_score.item()))
                #     print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

    fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
    # Apply the transformations needed
    import torchvision.transforms as T

    trf = T.Compose([T.Resize(imsize),
                     T.CenterCrop(imsize),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)  # Apply the transformations needed
    # inp = trf(cot_img).unsqueeze(0)# Apply the transformations needed

    out = fcn(inp)['out']

    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    def decode_segmap(image, nc=21):

        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (1, 1, 1),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    trf = T.Compose([T.Resize(256),
                     T.CenterCrop(256)])
    i = trf(img)
    rgb = torch.tensor(decode_segmap(om))
    i = torch.tensor(np.asarray(i))
    sty = output.squeeze(0)
    sty = sty.permute(1, 2, 0)
    a = rgb * sty
    a = a.detach()

    # Define the helper function
    def decode_segmap(image, nc=21):

        label_colors = np.array([(1, 1, 1),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (0, 0, 0),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

    trf = T.Compose([T.Resize(256),
                     T.CenterCrop(256)])

    rgb2 = torch.tensor(decode_segmap(om))
    i2 = torch.tensor(np.asarray(i))

    a2 = i2 * rgb2
    a2[a2 == 0] = 1
    a[a == 0] = 1 / 255

    ans = a2 * a

    trans = ans.numpy()
    aaa = np.transpose(trans, [0, 1, 2])

    x = aaa * 255
    x = Image.fromarray(x.astype(np.uint8))

    return x

@app.route("/", methods=['GET'])
def hello():
    return 'hello world'

@app.route("/predict", methods=['POST','GET'])
def predict():
    if request.method=='POST':
        data = request.get_json()
        dict_data = json.loads(data)

        content = dict_data['content_img']
        style = dict_data['style_img']

        img = preprocess(content, style)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_byte = buffered.getvalue()  # bytes
        img_base64 = base64.b64encode(img_byte)
        # まだbytesなのでjson.dumpsするためにstrに変換(jsonの要素はbytes型に対応していないため)
        img_str = img_base64.decode('utf-8')  # str
        result = {
            'data': img_str
        }
        return jsonify(result)
    else:
        data = request.get_json()
        return jsonify({'aaa':data})

if __name__ =='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
