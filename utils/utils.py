
import os 
from PIL import Image
import tqdm

def average_image_size():
    total_width = 0
    total_height = 0
    total = 0

    for i in os.listdir('data'):
        for j in tqdm.tqdm(os.listdir('data/' + i)):
            for k in os.listdir('data/' + i + '/' + j):
                total += 1
                try: 
                    img = Image.open('data/' + i + '/' + j + '/' + k)
                except:
                    print('data/' + i + '/' + j + '/' + k)
                    os.remove('data/' + i + '/' + j + '/' + k)
                width, height = img.size
                total_width += width
                total_height += height
    return (total_width / total, total_height / total)

print(average_image_size())