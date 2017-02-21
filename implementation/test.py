from scanreader import Scan
import matplotlib.pyplot as plt
from util import *

refscan = Scan("data/scan0")
errorscan = applytuple(refscan.scan_points, 0, 1, 0)

print hausdorff(errorscan, errorscan)

x = 0
y = 1
r = 0.5

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter([int(i[0]) for i in refscan.scan_points], [int(i[1]) for i in refscan.scan_points], s=10, c='b', marker='s', label='original')

ax1.scatter([int(i[0]) for i in errorscan], [int(i[1]) for i in errorscan], s=5, c='r', marker='x', label='error')

plt.legend(loc='upper left')
plt.show()
