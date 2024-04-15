from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import PIL
from PIL import Image
from matplotlib import colormaps
from torchvision.transforms.functional import to_pil_image
import numpy as np
from traintest import load_resnext_model
from adjustimage import AdjustImage
from torchvision.transforms import v2
import os
# Tutorial used for this class is: https://towardsdatascience.com/grad-cam-in-pytorch-use-of-forward-and-backward-hooks-7eba5e38d569

# defines two global scope variables to store our gradients and activations
gradients = None
activations = None


def backward_hook(module, grad_input, grad_output):
    global gradients  # refers to the variable in the global scope
    print('Backward hook running...')
    gradients = grad_output
    # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    print(f'Gradients size: {gradients[0].size()}')
    # We need the 0 index because the tensor containing the gradients comes
    # inside a one element tuple.


def forward_hook(module, args, output):
    global activations  # refers to the variable in the global scope
    print('Forward hook running...')
    activations = output
    # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    print(f'Activations size: {activations.size()}')


def remove_hooks(backward_hook_var, forward_hook_var):
    backward_hook_var.remove()
    forward_hook_var.remove()


def run_gradcam(model, image):
    # Accesses the relevant layer for ResNext model - if using other models, this would need to be changed
    final_layer = model.module.layer4[2]
    backward_hook_var = final_layer.register_full_backward_hook(
        backward_hook, prepend=False)
    forward_hook_var = final_layer.register_forward_hook(
        forward_hook, prepend=False)
    
    model(image.unsqueeze(0)).sum().backward()
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(2):
        # weight the channels by corresponding gradients
        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        heatmap = F.relu(heatmap)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        #plt.matshow(heatmap.detach().cpu())

    #Heatmap generated. Now overlay
    fig, ax = plt.subplots()
    ax.axis('off') # removes the axis markers

    # First plot the original image
    #This assumes the same bug seen in test (long tensor inverts image) is present
    ax.imshow(PIL.ImageOps.invert(to_pil_image(image, mode='RGB')))

    # Resize the heatmap to the same size as the input image and defines
    # a resample algorithm for increasing image resolution
    # we need heatmap.detach() because it can't be converted to numpy array while
    # requiring gradients
    
    # Suppressed error due to false positive in VSCode
    # pylint: disable=no-member
    overlay = to_pil_image(heatmap.detach(), mode='F').resize((256,256), resample=Image.BICUBIC)
    # pylint: enable=no-member
    # Apply any colormap you want
    cmap = colormaps['jet']
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

    # Plot the heatmap on the same axes,
    # but with alpha < 1 (this defines the transparency of the heatmap)
    ax.imshow(overlay, alpha=0.4, interpolation='nearest', extent=None)

    file_path = os.path.dirname(os.path.abspath(__file__))
    # Show the plot
    plt.savefig(str(Path(file_path + "/static/results/modelgradcamoutput.jpg")),bbox_inches='tight',pad_inches=0.0)
    #plt.show()
def main():
    DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
    CLASSIFICATION_TRANSFORMS = v2.Compose([v2.Resize([256,256]), v2.PILToTensor()])
    model = load_resnext_model(DEVICE,"models\\classification_model.pth" )
    image = Image.open("test_images\\0093_0597190523_02_WRI-R2_M012.jpg")
    image = CLASSIFICATION_TRANSFORMS(image) 
    
    image = image.type(torch.FloatTensor)
    run_gradcam(model, image)

if __name__ == "__main__":
    main()
