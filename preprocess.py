import os
from PIL import Image

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


img = './cover.png'
im = Image.open(img)
print(im)
new_size = (500, 200)
im = im.resize(new_size)

full_view = Image.new('RGB', (new_size[0]*4, new_size[1]))
""" 
for i in range(len([eval(i) for i in get_immediate_subdirectories('./imgs/')])):
    for j in range(4):
        img2 = './imgs/' + str(i) + '/' + str(j) + '.jpg'
        background = Image.open(img2)
        background = background.resize(new_size)
        background.paste(im, (0,0), im.convert('RGBA')) #(img to paste, coords, mask)
        full_view.paste(background, (j*new_size[0], 0))
    full_view.save('./preprocessed_imgs/' + str(i) + '.jpg')   """

# the name index of the picture it starts to preprocess
starting_index = 644
for i in range(len([eval(i) for i in get_immediate_subdirectories('./imgs/')])-500):
    for j in range(4):
        img2 = './imgs/' + str(i+starting_index) + '/' + str(j) + '.jpg'
        background = Image.open(img2)
        background = background.resize(new_size)
        background.paste(im, (0,0), im.convert('RGBA')) #(img to paste, coords, mask)
        full_view.paste(background, (j*new_size[0], 0))
    full_view.save('./preprocessed_imgs/' + str(i+starting_index) + '.jpg')    