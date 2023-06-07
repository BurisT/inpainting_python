import time
import matplotlib.pyplot as plt
import math
tic = time.time()
a=1
b=2
c= 30*9.9 + a + b
print(c)
plt.plot([1,2,math.sin(c)], [1,2,4])

toc = time.time()-tic

print('time = ', toc)

plt.show()