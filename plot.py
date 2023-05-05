import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import openpyxl

'''
x = np.linspace(-1, 1, 50)
x = np.array(x)
print(x)
y = np.linspace(0, 1, 50)
y = np.array(y)
c = x + y
print(c)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(x, y, c, cmap=plt.cm.brg)

plt.show()'''

'''
fig = plt.figure(figsize=(10, 10), facecolor='white')  
sub = fig.add_subplot(111, projection='3d')  

surf = sub.plot_surface(x_, y_, c, cmap=plt.cm.brg)  
cb = fig.colorbar(surf, shrink=0.3, aspect=16)  

sub.set_xlabel(r"x", fontsize=14, fontproperties='Calibri', fontstyle='italic')
sub.set_ylabel(r"t", fontsize=14, fontproperties='Calibri', fontstyle='italic')
sub.set_zlabel(r"u(x,t)", fontsize=14, fontproperties='Calibri', fontstyle='italic')

plt.title("Exact solution", fontproperties='Calibri', fontsize=18)
plt.show()
'''

# Load the workbook
workbook = openpyxl.load_workbook('solution.xlsx')
# Select the worksheet
worksheet = workbook['train']
# Create empty lists to store the data
list1 = []
list2 = []
list3 = []
# Loop through the rows in the worksheet and append the data to the appropriate list
for row in worksheet.iter_rows(values_only=True):
    list1.append(row[0])
    list2.append(row[1])
    list3.append(row[2])


X = list1
Y = list2
Z = list3
'''# Exact solution
fig = plt.figure(figsize=(10, 10), facecolor='white')
ax = fig.add_subplot(111, projection='3d')
x1 = np.array(list1)
y1 = np.array(list2)
c = np.exp(-y1) * np.sin(np.pi * x1)
sub = fig.add_subplot(111, projection='3d')
surf = sub.plot_trisurf(X, Y, Z, cmap=plt.cm.brg)
cb = fig.colorbar(surf, shrink=0.5, aspect=18)
sub.set_xlabel(r"x", fontsize=14, fontproperties='Calibri', fontstyle='italic')
sub.set_ylabel(r"t", fontsize=14, fontproperties='Calibri', fontstyle='italic')
sub.set_zlabel(r"u_pre", fontsize=14, fontproperties='Calibri', fontstyle='italic')
sub.set_title("Approximate solution", fontproperties='Calibri', fontsize=16)
plt.savefig("Approximate solution", dpi=500)
plt.show()
'''
x1 = np.array(list1)
y1 = np.array(list2)

c = np.exp(-y1) * np.sin(np.pi * x1)
fig = plt.figure(figsize=(10, 10), facecolor='white')
ax = fig.add_subplot(111, projection='3d')
# Plot the surface
surf = ax.plot_trisurf(X, Y, Z, cmap=plt.cm.brg)
cb = fig.colorbar(surf, shrink=0.3, aspect=18)
ax.set_xlabel(r"x", fontsize=16, fontproperties='Calibri', fontstyle='italic')
ax.set_ylabel(r"t", fontsize=16, fontproperties='Calibri', fontstyle='italic')
ax.set_zlabel(r"u", fontsize=16, fontproperties='Calibri', fontstyle='italic')
ax.set_title("Approximate solution", fontproperties='Calibri', fontsize=20)
# plt.title("Approximate solution", fontproperties='Calibri', fontsize=16)
plt.savefig("Approximate solution.png", dpi=400)
plt.show()
# Error
'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(X, Y, c - Z, cmap='rainbow')
ax.set_xlabel(r"x", fontsize=12, fontproperties='Calibri', fontstyle='italic')
ax.set_ylabel(r"t", fontsize=12, fontproperties='Calibri', fontstyle='italic')
ax.set_zlabel(r"error", fontsize=12, fontproperties='Calibri', fontstyle='italic')
plt.title("Absolute error", fontproperties='Calibri', fontsize=14)
plt.savefig("absolute_error.png", dpi=500)
plt.show()
print("max:", max(abs(c - Z)))
print("min:", min(abs(c - Z)))'''