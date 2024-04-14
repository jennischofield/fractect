from torch.autograd import Variable
import torch
import torch.nn.functional as nnf


def get_conf_for_image(image, model, device):

    image = image.to(device)
    test = Variable(image)
    output = model(test.type(torch.FloatTensor))
    prob = nnf.softmax(output, dim=1)
    top_p, top_class = prob.topk(2, dim=1)
    # 0 - fractured, 1 - not fractured
    return zip(top_p.cpu().detach().numpy().tolist(), top_class.cpu().detach().numpy().tolist())
