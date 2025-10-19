import cv2
import numpy as np
import matplotlib.pyplot as plt

def ghe(img): # global histogram equalization
	img_his, img_bin = np.histogram(img.flatten(), 256, [0, 256])
	img_cdf = img_his.cumsum()
	img_cdf_norm = (img_cdf - img_cdf.min()) * 255 / (img_cdf.max() - img_cdf.min())
	img_eql = (img_cdf_norm[img]).astype(np.uint8)
	res = img_eql
	return res

def gen_mask(img, k): # generate mask
	k = k // 2
	sigma = 2
	x, y = np.mgrid[-k:k+1, -k:k+1]
	mask = np.exp(-(x**2 + y**2) / (2 * sigma**2))
	mask = mask / mask.sum()
	return mask

def conv(img, mask): # convolution
	row, col = img.shape
	pad_row, pad_col = mask.shape[0] // 2, mask.shape[1] // 2

	pad_img = np.pad(img, (pad_row, pad_col), mode='reflect')
	flt_img = np.zeros_like(img)

	for i in range(row):
		for j in range(col):
			cur = pad_img[i:i + 2 * pad_row + 1, j:j + 2 * pad_col + 1]
			flt_img[i, j] = np.sum(cur * mask)
	return flt_img.astype(np.uint8)

def medf(img, mask_size): # median filtering
	row, col = img.shape
	pad_size = mask_size // 2

	pad_img = np.pad(img, (pad_size, pad_size), mode='reflect')
	flt_img = np.zeros_like(img)

	for i in range(row):
		for j in range(col):
			cur = pad_img[i:i + 2 * pad_size + 1, j:j + 2 * pad_size + 1]
			flt_img[i, j] = np.median(cur)
	return flt_img.astype(np.uint8)

def PSNR(img_o, img_c):
	mse = np.mean((img_o - img_c)**2)
	if mse == 0:
		return float('inf')
	pxl = 255**2
	psnr = 10 * np.log10(pxl / mse)
	return psnr

# Prob 0.(a)
smp1 = cv2.imread("hw1_sample_images/sample1.png")

flp1 = smp1[:, ::-1, :] # column switch

height, width, channels = flp1.shape

cmb1 = np.zeros((height, width * 2, channels), dtype=np.uint8)

cmb1[:, :width, :] = smp1 # combine
cmb1[:, width:, :] = flp1 # combine
res1 = cmb1

cv2.imwrite("result1.png", res1)

# Prob 0.(b)
B = res1[:, :, 0]
G = res1[:, :, 1]
R = res1[:, :, 2]

gray = (B * 0.114 + G * 0.587 + R * 0.299).astype(np.uint8)
res2 = gray

cv2.imwrite("result2.png", res2)

# Prob 1.(a)
smp2 = cv2.imread("hw1_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE)

dark = smp2 // 3

res3 = dark
cv2.imwrite("result3.png", res3)

# Prob 1.(b)
lght = res3 * 3
res4 = lght
cv2.imwrite("result4.png", res4)

# Prob 1.(c)
plt.figure(figsize=(15, 5)) # plot histogram

plt.subplot(1, 3, 1)
plt.hist(smp2.flatten(), 256, [0, 256])
plt.title("Histogram of sample2.png")

plt.subplot(1, 3, 2)
plt.hist(res3.flatten(), 256, [0, 256])
plt.title("Histogram of result3.png")

plt.subplot(1, 3, 3)
plt.hist(res4.flatten(), 256, [0, 256])
plt.title("Histogram of result4.png")

plt.show()

# Prob 1.(d)
res5 = ghe(smp2) # global histogram equalization
cv2.imwrite("result5.png", res5)

res6 = ghe(res3)
cv2.imwrite("result6.png", res6)

res7 = ghe(res4)
cv2.imwrite("result7.png", res7)

plt.figure(figsize=(15, 5)) # plot histogram

plt.subplot(1, 3, 1)
plt.hist(res5.flatten(), 256, [0, 256])
plt.title("Histogram of result5.png")

plt.subplot(1, 3, 2)
plt.hist(res6.flatten(), 256, [0, 256])
plt.title("Histogram of result6.png")

plt.subplot(1, 3, 3)
plt.hist(res7.flatten(), 256, [0, 256])
plt.title("Histogram of result7.png")

plt.show()

# Prob 1.(e)
block_size = 32
overlap = block_size // 2 # overlapping
(row, col) = smp2.shape

smp2_lhe = np.zeros_like(smp2, dtype=np.float32)
wght_sum = np.zeros_like(smp2, dtype=np.float32)

for i in range(0, row, overlap):
    for j in range(0, col, overlap):
        i_end = min(i + block_size, row)
        j_end = min(j + block_size, col)
        block = smp2[i:i_end, j:j_end]
        
        eql_block = ghe(block) # global histogram equalization
        
		# bilinear interpolation
        wght_i, wght_j = np.mgrid[0:block.shape[0], 0:block.shape[1]]
        wght = (1 - np.abs(wght_j - block.shape[1] / 2) / (block.shape[1] / 2)) * (1 - np.abs(wght_i - block.shape[0] / 2) / (block.shape[0] / 2))
        
        smp2_lhe[i:i_end, j:j_end] += eql_block * wght
        wght_sum[i:i_end, j:j_end] += wght

smp2_lhe = np.divide(smp2_lhe, wght_sum, where=wght_sum != 0)
smp2_lhe = np.clip(smp2_lhe, 0, 255).astype(np.uint8)
res8 = smp2_lhe

cv2.imwrite("result8.png", res8)

plt.figure(figsize=(10, 5)) # plot histogram

plt.subplot(1, 2, 1)
plt.hist(smp2.flatten(), 256, [0, 256])
plt.title("Histogram of sample2.png")

plt.subplot(1, 2, 2)
plt.hist(res8.flatten(), 256, [0, 256])
plt.title("Histogram of result8.png")

plt.show()

# Prob 1.(f)
smp3 = cv2.imread("hw1_sample_images/sample3.png", cv2.IMREAD_GRAYSCALE)

res9 = np.power((smp3 / 255), 0.3) * 255 # Power-Law

cv2.imwrite("result9.png", res9)

plt.figure(figsize=(10, 5)) # plot histogram

plt.subplot(1, 2, 1)
plt.hist(smp3.flatten(), 256, [0, 256])
plt.title("Histogram of sample3.png")

plt.subplot(1, 2, 2)
plt.hist(res9.flatten(), 256, [0, 256])
plt.title("Histogram of result9.png")

plt.show()

# Prob 2.(a)
smp5 = cv2.imread("hw1_sample_images/sample5.png", cv2.IMREAD_GRAYSCALE)
mask5 = gen_mask(smp5, 9) # get mask
res10 = conv(smp5, mask5) # convolution
cv2.imwrite("result10.png", res10)

smp6 = cv2.imread("hw1_sample_images/sample6.png", cv2.IMREAD_GRAYSCALE)
res11 = medf(smp6, 3) # Median
cv2.imwrite("result11.png", res11)

# Prob 2.(b)
smp4 = cv2.imread("hw1_sample_images/sample4.png", cv2.IMREAD_GRAYSCALE)
smp4 = smp4.astype(float)
res10 = res10.astype(float)
res11 = res11.astype(float)
print(f"PSNR values of result10.png is {PSNR(smp4, res10)}")
print(f"PSNR values of result11.png is {PSNR(smp4, res11)}")

# Prob 2.(c)
smp7 = cv2.imread("hw1_sample_images/sample7.png", cv2.IMREAD_GRAYSCALE)

res12 = np.power((smp7 / 255), 0.6) * 255 # Power-Law
res12 = medf(res12, 3) # Median
mask12 = gen_mask(res12, 3) # Low-pass
res12 = conv(res12, mask12) # Low-pass

cv2.imwrite("result12.png", res12)
print(f"PSNR values of result12.png is {PSNR(smp4, res12)}")
