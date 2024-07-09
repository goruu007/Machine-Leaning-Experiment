import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
img = cv2.imread('C:\\Users\\gaurav raikwar\\OneDrive\\Documents\\Desktop\\New folder\edit.jpeg')
a = np.array(img.data)
maximum = a.max()
b = maximum-a
plt.subplot(1,2,1)
plt.title('ORIGINAL')
plt.imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2)
plt.title('NEGATION')
plt.imshow(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
plt.show()
