from PIL import Image

def prepare_screenshot(path, target_width=800):
    img = Image.open(path)
    width,height = img.size

    width_new,height_new = (target_width, round(height*(target_width/width)))
    img = img.resize((width_new, height_new))
    img = img.convert('L')

    return img




if __name__ == '__main__':
    small_img = prepare_screenshot('t.jpg')
    small_img400 = prepare_screenshot('t.jpg', target_width=400)

    small_img.save('resulte.png')

    print(small_img.size)
    print(small_img400.size)



