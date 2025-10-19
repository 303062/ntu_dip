import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Prob. 1 (a)
# sobel gradient image
def sobel_edge_dect(img):
	rol, col = img.shape
	gx = np.array([[1, 0 ,-1], [2, 0, -2], [1, 0, -1]])
	gy = np.array([[-1, -2, -1], [0, 0 ,0], [1, 2, 1]])
	mask_size = 3
	pad = mask_size // 2
	pad_img = np.pad(img, (pad, pad), mode="reflect")
	res = np.zeros_like(img, dtype=np.float32)
	for i in range(rol):
		for j in range(col):
			reg = pad_img[i:i+mask_size, j:j+mask_size]
			x = np.sum(gx * reg)
			y = np.sum(gy * reg)
			res[i, j] = np.sqrt(x**2 + y**2)

	res = (res / res.max() * 255).astype(np.uint8)

	return res

# sobel edge map
def sobel_edge_map(img, thr):
	res = img.copy()
	res[img > thr] = 255
	res[img <= thr] = 0
	return res

smp1 = cv2.imread("hw2_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
res1 = sobel_edge_dect(smp1)
cv2.imwrite("result1.png", res1)
thr1 = 30
res2 = sobel_edge_map(res1, thr1)
cv2.imwrite("result2.png", res2)

# Prob. 1 (b)
def gen_mask(k): # generate mask
	k = k // 2
	sigma = 1.4
	x, y = np.mgrid[-k:k+1, -k:k+1]
	mask = np.exp(-(x**2 + y**2) / (2 * sigma**2))
	mask = mask / mask.sum()
	return mask

def conv(img, mask): # convolution
	row, col = img.shape
	pad_row, pad_col = mask.shape[0] // 2, mask.shape[1] // 2

	pad_img = np.pad(img, (pad_row, pad_col), mode='reflect')
	flt_img = np.zeros_like(img, dtype="float32")

	for i in range(row):
		for j in range(col):
			cur = pad_img[i:i + 2 * pad_row + 1, j:j + 2 * pad_col + 1]
			flt_img[i, j] = np.sum(cur * mask)
	return flt_img

# compute gradient magnitude and orientation
def grad_mag_ori(img):
	rol, col = img.shape
	img = img.astype(np.uint8)
	gx = np.array([[1, 0 ,-1], [2, 0, -2], [1, 0, -1]])
	gy = np.array([[-1, -2, -1], [0, 0 ,0], [1, 2, 1]])
	mask_size = 3
	pad = mask_size // 2
	pad_img = np.pad(img, (pad, pad), mode="reflect")
	mag = np.zeros_like(img, dtype=np.float32)
	ori = np.zeros_like(img, dtype=np.float32)
	for i in range(rol):
		for j in range(col):
			reg = pad_img[i:i + 2 * pad + 1, j:j + 2 * pad + 1]
			x = np.sum(gx * reg)
			y = np.sum(gy * reg)
			mag[i, j] = np.sqrt(x**2 + y**2)
			ori[i, j] = np.arctan2(y, x)

	return mag, ori

# non-maximal suppression
def non_max_sup(mag, ori):
	row, col = mag.shape
	res = np.zeros_like(mag, dtype=np.float32)
	ang = ori * 180 / np.pi
	ang[ang < 0] += 180
	for i in range(1, row - 1):
		for j in range(1, col - 1):
			x = 255
			y = 255
			if (0 <= ang[i, j] < 22.5) or (157.5 <= ang[i, j] <= 180):
				x = mag[i, j + 1]
				y = mag[i, j - 1]
			elif 22.5 <= ang[i, j] < 67.5:
				x = mag[i + 1, j - 1]
				y = mag[i - 1, j + 1]
			elif 67.5 <= ang[i, j] < 112.5:
				x = mag[i + 1, j]
				y = mag[i - 1, j]
			elif 112.5 <= ang[i, j] < 157.5:
				x = mag[i - 1, j - 1]
				y = mag[i + 1, j + 1]

			if (mag[i, j] >= x) and (mag[i, j] >= y):
				res[i, j] = mag[i, j]
			else:
				res[i, j] = 0
	return res

# hysteretic thresholding
def hys_thr(img, hgh, low, strg, weak):
	si, sj = np.where(img >= hgh)
	wi, wj = np.where((img < hgh) & (img >= low))
	res = np.zeros_like(img, dtype=np.uint8)
	res[si, sj] = strg
	res[wi, wj] = weak

	return res

# connected component labeling method
def edge_track(img, strg, weak):
	row, col = img.shape
	for i in range(row):
		for j in range(col):
			if img[i, j] == weak:
				if (img[i - 1, j - 1] == strg or img[i - 1, j] == strg or img[i - 1, j + 1] == strg or img[i, j - 1] == strg or img[i, j + 1] == strg or img[i + 1, j - 1] == strg or img[i + 1, j] == strg or img[i + 1, j + 1] == strg):
					img[i, j] = strg
				else:
					img[i, j] = 0
	return img

k = 5
mask3 = gen_mask(k)
tmp3 = conv(smp1, mask3)
mag3, ori3 = grad_mag_ori(tmp3)
nms3 = non_max_sup(mag3, ori3)
hgh = 60
low = 30
strg = 255
weak = 1
thr3 = hys_thr(nms3, hgh, low, strg, weak)
res3 = edge_track(thr3, strg, weak)
cv2.imwrite("result3.png", res3)

# Prob. 1 (c)
# set up a threshold to separate zero and non-zero
def thr_sep(img, thr):
	row, col = img.shape
	thr_img = np.zeros_like(img)
	for i in range(1, row - 1):
		for j in range(1, col -1 ):
			if np.abs(img[i, j]) <= thr:
				thr_img[i, j] = 255
			else:
				thr_img[i, j] = 0
	return thr_img

# decide zero-crossing points
def zero_cross(lap, thr):
	row, col = lap.shape
	res = np.zeros_like(lap)
	for i in range(1, row - 1):
		for j in range(1, col - 1):
			if thr[i, j] == 0:
				if (lap[i - 1, j - 1] * lap[i + 1, j + 1] < 0) or (lap[i, j + 1] * lap[i, j - 1] < 0) or (lap[i - 1, j + 1] * lap[i + 1, j - 1] < 0) or (lap[i + 1, j] * lap[i - 1, j] < 0):
					res[i, j] = 255
				else:
					res[i, j] = 0
			else:
				res[i, j] = 0
	return res

mask4 = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]) * (1 / 273)
blur4 = conv(smp1, mask4)
lap_mask = np.array([[-2, 1, -2], [1, 4, 1], [-2, 1, -2]]) * (1 / 8)
lap4 = conv(blur4, lap_mask)
thr_val = 2
thr4 = thr_sep(lap4, thr_val)
res4 = zero_cross(lap4, thr4)
res4 = np.uint8(res4)
cv2.imwrite("result4.png", res4)

# Prob. 1 (d)
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

resd = medf(smp1, 17)
cv2.imwrite("resultd.png", resd)

# Prob. 1 (e)
# get hough space
def get_hough_space(img):
	row, col = img.shape
	max_rho = int(round(math.sqrt(row**2 + col**2)))
	rhos = np.arange(-max_rho, max_rho, 1)
	thetas = np.deg2rad(np.arange(-90, 90, 1))
	acc = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
	edg_pts = np.argwhere(img)
	for y, x in edg_pts:
		for theta_idx, theta in enumerate(thetas):
			rho = int(x * np.cos(theta) + y * np.sin(theta))
			rho_idx = np.argmin(np.abs(rhos - rho))
			acc[rho_idx, theta_idx] += 1
	return acc, rhos, thetas

# plot hough space
def plot_hough_space(acc, rhos, res):
	plt.imshow(acc, aspect='auto', cmap='gray', extent=[-90, 90, -len(rhos)//2, len(rhos)//2])
	plt.xlabel('Theta (degrees)')
	plt.ylabel('Rho (pixels)')
	plt.title('Hough Transform Space')
	plt.colorbar(label='Votes')
	plt.savefig(res)
	return

def get_lines(acc, rhos, thetas, threshold):
	lines = []
	for rho_idx, theta_idx in np.argwhere(acc > threshold):
		rho = rhos[rho_idx]
		theta = thetas[theta_idx]
		lines.append((rho, theta))
	return np.array(lines)

# overlay result7.png on result5.png  
def overlay(img, lines):
	if lines is not None:
		for rho, theta in lines:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a * rho
			y0 = b * rho
			x1 = int(x0 + 1000 * (-b))
			y1 = int(y0 + 1000 * (a))
			x2 = int(x0 - 1000 * (-b))
			y2 = int(y0 - 1000 * (a))
			img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
	return img

smp2 = cv2.imread("hw2_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE)
k5 = 5
mask5 = gen_mask(k5)
tmp5 = conv(smp2, mask5)
mag5, ori5 = grad_mag_ori(tmp5)
nms5 = non_max_sup(mag5, ori5)
hgh5 = 100
low5 = 50
strg5 = 255
weak5 = 1
thr5 = hys_thr(nms5, hgh5, low5, strg5, weak5)
res5 = edge_track(thr5, strg5, weak5)
cv2.imwrite("result5.png", res5)

acc5, rhos5, thetas5 = get_hough_space(res5)
plot_hough_space(acc5, rhos5, "result6.png")
thr5 = 100
lines5 = get_lines(acc5, rhos5, thetas5, thr5)
res7 = np.zeros((res5.shape[0], res5.shape[1], 3), dtype=np.uint8)
res7[:, :, 0] = res5
res7[:, :, 1] = res5
res7[:, :, 2] = res5
res7 = overlay(res7, lines5)
cv2.imwrite("result7.png", res7)

# Prob. 2 (a)
def man_remap(img, xmap, ymap):
	row, col = img.shape
	res = np.zeros_like(img)
	for y in range(row):
		for x in range(col):
			map_x = xmap[y, x]
			map_y = ymap[y, x]
			if 0 <= map_x < col - 1 and 0 <= map_y < row - 1:
				x0, y0 = int(map_x), int(map_y)
				x1, y1 = x0 + 1, y0 + 1
				dx, dy = map_x - x0, map_y - y0
				w1, w2 = 1 - dx, dx
				w3, w4 = 1 - dy, dy
				res[y, x] = (img[y0, x0] * w1 * w3 + img[y0, x1] * w2 * w3 + img[y1, x0] * w1 * w4 + img[y1, x1] * w2 * w4)
			else:
				res[y, x] = 255
	return res

# barrel distortion
def barrel_distort(img, xscl, yscl, xctr, yctr, rad, pwr):
	row, col = img.shape
	xmap = np.zeros((row, col), np.float32)
	ymap = np.zeros((row, col), np.float32)

	for y in range(row):
		for x in range(col):
			dy = yscl * (y - yctr)
			dx = xscl * (x - xctr)
			dist = dx**2 + dy**2
			if dist >= rad**2:
				xmap[y, x] = x
				ymap[y, x] = y
			else:
				fact = 1
				if dist > 0:
					fact = math.pow(math.sin(math.pi * math.sqrt(dist) / rad / 2), pwr)
				xmap[y, x] = fact * dx / xscl + xctr
				ymap[y, x] = fact * dy / yscl + yctr
	res = man_remap(img, xmap, ymap)
	return res

def translate(img, dx, dy):
	res = np.zeros_like(img)
	row, col = img.shape
	for y in range(row):
		for x in range(col):
			v = y + dy
			u = x - dx
			if 0 <= v < row and 0 <= u < col:
				res[y, x] = img[v, u]
			else:
				res[y, x] = 255
	return res

def rotate(img, theta):
	res = np.zeros_like(img)
	row, col = img.shape
	mid = row / 2
	co = np.cos(theta * np.pi / 180)
	si = np.sin(theta * np.pi / 180)
	for y in range(row):
		for x in range(col):
			v = int((y - mid) * co - (x - mid) * si + mid)
			u = int((y - mid) * si + (x - mid) * co + mid)
			if 0 <= v < row and 0 <= u < col:
				res[y, x] = img[v, u]
			else:
				res[y, x] = 255
	return res

def scale(img, scl):
	res = np.zeros_like(img)
	row, col = img.shape
	yctr, xctr = row / 2, col / 2

	for y in range(row):
		for x in range(col):
			v = int((y - yctr) / scl + yctr)
			u = int((x - xctr) / scl + xctr)	
			if 0 <= v < row and 0 <= u < col:
				res[y, x] = img[v, u]
			else:
				res[y, x] = 255
	
	return res

smp3 = cv2.imread("hw2_sample_images/sample3.png", cv2.IMREAD_GRAYSCALE)
xscl6, yscl6 = 1, 1
xctr6, yctr6 = 300, 320
rad6 = smp3.shape[0] // 2
pwr6 = 1
res8 = barrel_distort(smp3, xscl6, yscl6, xctr6, yctr6, rad6, pwr6)
res8 = scale(res8, 0.8)
res8 = rotate(res8, -15)
res8 = translate(res8, -75, 75)
cv2.imwrite("result8.png", res8)

# Prob. 2 (b)
def cos_vortex(img, ctr, rad, pwr):
	row, col = img.shape
	col_vec = np.arange(row)[:, None]
	row_vec = np.arange(col)[None, :]
	ctr_col_vec, ctr_row_vec = col_vec - ctr[1], row_vec - ctr[0]
	dist = np.sqrt(ctr_col_vec**2 + ctr_row_vec**2)
	thetas = np.arctan2(ctr_col_vec, ctr_row_vec) + pwr * np.exp(-dist / rad)
	fin_col = np.clip(np.int32(ctr[1] + dist * np.sin(thetas)), 0, row - 1)
	fin_row = np.clip(np.int32(ctr[0] + dist * np.cos(thetas)), 0, col - 1)
	return img[fin_col, fin_row]

smp5 = cv2.imread("hw2_sample_images/sample5.png", cv2.IMREAD_GRAYSCALE)
ctr7 = (200, 200)
rad7 = 40
pwr7 = 30
res9 = cos_vortex(smp5, ctr7, rad7, pwr7)
cv2.imwrite("result9.png", res9)