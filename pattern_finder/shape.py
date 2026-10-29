import numpy as np

from preprocess import  prepare_screenshot

def extract_line(pixels):  
    height,width = pixels.shape
    ys = []
    for col in range(width):
        column = pixels[:,col]
        row =  np.where(column < 128)[0]
        if len(row) > 0:
            ys.append(height - 1 - row.mean())  
    return ys

def normalize(values):
    arr = np.array(values)
    if arr.max() == arr.min():
        return np.zeros(len(arr))
    return (arr -arr.min())/((arr -arr.min()).max())

def resample(values, n):
    old_x = np.linspace(0, 1, len(values))
    new_x = np.linspace(0, 1, n)
    return np.interp(new_x, old_x, values)

def photo_to_shape(path, n=50):
    img = prepare_screenshot(path)   # серая картинка шириной 800
    pixels =np.array(img)               # картинка → таблица чисел
    ys =extract_line(pixels)                  # таблица → высоты линии
    ys = resample(ys, n)                  # 360+ точек → ровно n точек
    return normalize(ys)



if __name__ == "__main__":
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt

    print(normalize([100, 200, 300]))        # [0.  0.5 1. ]
    print(normalize([79000, 79500, 80000]))  # [0.  0.5 1. ]
    print(normalize([5, 5, 5]))              # [0. 0. 0.]

    img = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(img)
    points = [(20, 40), (110, 160), (200, 80), (290, 150), (380, 30)]
    draw.line(points, fill="black", width=3)
    img.save("w_shape.png")

    shape = photo_to_shape("w_shape.png")
    print(len(shape), shape.min(), shape.max())   # 50 0.0 1.0

    plt.plot(shape, marker="o")
    plt.savefig("shape.png")
    plt.close()