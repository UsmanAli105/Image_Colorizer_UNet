import os
from torchvision.io import read_image
import sys


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: <arg1>")
        sys.exit(0)

    path = sys.argv[1]
    if not os.path.exists(path):
        print('Path does not exists')
        sys.exit(0)

    images = os.listdir(path)

    for image in images:
        try:
            if read_image(os.path.join(path, image)).shape[0] != 3:
                print(image)
                os.remove(os.path.join(path, image))
        except Exception as e:
            print(image)
            os.remove(os.path.join(path, image))