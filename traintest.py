from torchvision.transforms import v2
from adjustimage import AdjustImage
def main():
    transforms = v2.Compose([AdjustImage(), v2.Resize([256,256]), v2.PILToTensor()])
    
if __name__ == "__main__":
    main()
