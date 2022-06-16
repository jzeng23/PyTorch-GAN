import os
from PIL import Image
from numpy import asarray

dir = 'data/raw'
i = 0
for img_name in os.listdir(dir):

    left = 392
    top = 0
    right = left + 1000
    bottom = top + 1000

    window = (left, top, right, bottom)
    img = Image.open(os.path.join(dir, img_name)).convert('L')

    # resulting dataset has size 45*34

    while bottom < img.size[1]:
        while right < 3010:
            window = (left, top, right, bottom)
            cropped = img.crop(window).resize((85, 85))
            filename = 'images/size85/data_' + str(i) + '.png'
            cropped.save(filename)
            left += 50
            right += 50
            i += 1
        top += 50
        bottom += 50