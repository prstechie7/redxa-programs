from PIL import Image
from torchvision import transforms


IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


class ResizePad:

    def __init__(self, size=IMAGE_SIZE):
        self.size = size

    def __call__(self, image):

        width, height = image.size

        scale = min(
            self.size / width,
            self.size / height
        )

        new_width = max(
            1,
            round(width * scale)
        )

        new_height = max(
            1,
            round(height * scale)
        )

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            "RGB",
            (self.size, self.size),
            (0, 0, 0)
        )

        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2

        canvas.paste(
            image,
            (left, top)
        )

        return canvas


def get_inference_transform():

    return transforms.Compose([
        ResizePad(IMAGE_SIZE),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])