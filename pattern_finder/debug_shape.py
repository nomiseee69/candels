import matplotlib.pyplot as plt

from shape import photo_to_shape

PATH = "user_813151593.jpg"

shape = photo_to_shape(PATH)
plt.plot(shape, marker="o")
plt.savefig("debug_shape.png")
plt.close()
print("Готово, открой debug_shape.png")