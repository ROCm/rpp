import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

for i in range(0,7):
    fileName = str(i) + ".csv"
    with open(fileName, 'rb+') as filehandle:
        filehandle.seek(-1, os.SEEK_END)
        filehandle.truncate()

u8u8 = np.genfromtxt('0.csv', delimiter=',')
f16f16 = np.genfromtxt('1.csv', delimiter=',')
f32f32 = np.genfromtxt('2.csv', delimiter=',')
u8f16 = np.genfromtxt('3.csv', delimiter=',')
u8f32 = np.genfromtxt('4.csv', delimiter=',')
i8i8 = np.genfromtxt('5.csv', delimiter=',')
u8i8 = np.genfromtxt('6.csv', delimiter=',')

u8u8 = u8u8 / 255
i8i8 = (i8i8 + 128) / 255
u8i8 = (u8i8 + 128) / 255

def rmse(prediction, target):
    return np.sqrt(((prediction - target) ** 2).mean())

rmse1 = rmse(f16f16, u8u8)
rmse2 = rmse(f32f32, u8u8)
rmse3 = rmse(u8f16, u8u8)
rmse4 = rmse(u8f32, u8u8)
rmse5 = rmse(i8i8, u8u8)
rmse6 = rmse(u8i8, u8u8)

print("Average Tensor Diff RMSE(f16f16 - u8u8) = ", rmse1)
print("Average Tensor Diff RMSE(f32f32 - u8u8) = ", rmse2)
print("Average Tensor Diff RMSE(u8f16 - u8u8) = ", rmse3)
print("Average Tensor Diff RMSE(u8f32 - u8u8) = ", rmse4)
print("Average Tensor Diff RMSE(i8i8 - u8u8) = ", rmse5)
print("Average Tensor Diff RMSE(u8i8 - u8u8) = ", rmse6)
