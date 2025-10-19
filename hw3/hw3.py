import cv2
import numpy as np
import matplotlib.pyplot as plt

# Problem 1. (a)
# Erosion
def erode(img, mask):
	rows, cols = img.shape
	pad_rows, pad_cols = mask.shape[0] // 2, mask.shape[1] // 2
	res = img.copy()
	tmp = img.copy()
	for i in range(pad_rows, rows - pad_rows):
		for j in range(pad_cols, cols - pad_cols):
			reg = tmp[i - pad_rows : i + pad_rows + 1, j - pad_cols : j + pad_cols + 1]
			if np.all((reg & mask) == mask):
				res[i, j] = 255
			else:
				res[i, j] = 0
	return res

# Dilation
def dilate(img, mask):
	rows, cols = img.shape
	pad_rows, pad_cols = mask.shape[0] // 2, mask.shape[1] // 2
	res = img.copy()
	tmp = img.copy()
	for i in range(pad_rows, rows - pad_rows):
		for j in range(pad_cols, cols - pad_cols):
			reg = tmp[i - pad_rows : i + pad_rows + 1, j - pad_cols : j + pad_cols + 1]
			if np.any(reg & mask):
				res[i, j] = 255
			else:
				res[i, j] = 0
	return res

smp1 = cv2.imread("hw3_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
mask1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
eroded = erode(smp1, mask1)
res1 = dilate(eroded, mask1)
cv2.imwrite("result1.png", res1)

# Problem 1. (b)
def flood_fill1(img, i, j):
	rows, cols = img.shape
	res = img.copy()
	mask = np.zeros_like(img, dtype=np.uint8)
	queue = [(i, j)]
	while queue:
		ci, cj = queue.pop(0)
		if mask[ci, cj] == 0 and img[ci, cj] == 0:
			res[ci, cj] = 255
			mask[ci, cj] = 1
			queue.append((ci - 1, cj))
			queue.append((ci + 1, cj))
			queue.append((ci, cj - 1))
			queue.append((ci, cj + 1))
	return res

def inv(img):
	rows, cols = img.shape
	res = np.zeros_like(img, dtype=np.uint8)
	for i in range(rows):
		for j in range(cols):
			res[i, j] = 255 - img[i, j]
	return res

def hole_fill(img):
	tmp = img.copy()
	tmp = flood_fill1(tmp, 0, 0)
	tmp_inv = inv(tmp)
	res = img | tmp_inv
	return res
res2 = hole_fill(res1)
cv2.imwrite("result2.png", res2)

# Problem 1. (c)
# Zhang-Suen Thinning Algorithm
def skelet(img):
	rows, cols = img.shape
	skt = img.copy()
	done = False
	while not done:
		done = True
		discard = []

		for pas in range(2):
			for i in range(1, rows - 1):
				for j in range(1, cols - 1):
					if skt[i, j] == 255:
						adj = [skt[i - 1, j], skt[i - 1, j + 1], skt[i, j + 1], skt[i + 1, j + 1], skt[i + 1, j], skt[i + 1, j - 1], skt[i, j - 1], skt[i - 1, j - 1]]
						adj = [int(k) for k in adj]
						num_white = sum(k == 255 for k in adj)
						num_trans = sum((adj[k] == 0 and adj[k + 1] == 255) for k in range(7)) + (adj[7] == 0 and adj[0] == 255)

						if pas == 0:
							if 1 < num_white < 7 and num_trans == 1 and (adj[0] * adj[2] * adj[4] == 0) and (adj[2] * adj[4] * adj[6] == 0):
								discard.append((i, j))
						
						if pas == 1:
							if 1 < num_white < 7 and num_trans == 1 and (adj[0] * adj[2] * adj[6] == 0) and (adj[0] * adj[4] * adj[6] == 0):
								discard.append((i, j))

			for i, j in discard:
				skt[i, j] = 0
				done = False

			discard = []
	
	return skt

res3 = skelet(res1)
cv2.imwrite("result3.png", res3)

# Problem 1. (d)
def flood_fill2(img, labs, i, j, lab):
	rows, cols = img.shape
	stack = [(i, j)]
	while stack:
		ci, cj = stack.pop()
		if labs[ci, cj] == 0 and img[ci, cj] == 255:
			labs[ci, cj] = lab
			stack.append((ci - 1, cj))
			stack.append((ci + 1, cj))
			stack.append((ci, cj - 1))
			stack.append((ci, cj + 1))
	return


# get labels and count objects
def cnt_comp(img):
	rows, cols = img.shape
	labs = np.zeros_like(img, dtype=np.uint8)
	lab = 1
	for i in range(rows):
		for j in range(cols):
			if img[i, j] == 255 and labs[i, j] == 0:
				flood_fill2(img, labs, i, j, lab)
				lab += 1
	cnt = lab - 1
	return cnt, labs

_, labsd = cnt_comp(res2)
tmpd = np.isin(labsd, [8, 16]).astype(np.uint8) * 255
maskd = np.ones((5, 5), np.uint8)
dltd = dilate(tmpd, maskd)
resd = res2.copy()
resd[dltd == 255] = 255
cv2.imwrite("res1d.png", resd)
cntd, _ = cnt_comp(resd)
print(f"The number of objects in sample1.png: {cntd}")

# Problem 2. (a)
def conv(img, mask): # convolution
	row, col = img.shape
	pad_row, pad_col = mask.shape[0] // 2, mask.shape[1] // 2

	pad_img = np.pad(img, (pad_row, pad_col), mode='reflect')
	flt_img = np.zeros_like(img)

	for i in range(row):
		for j in range(col):
			cur = pad_img[i:i + 2 * pad_row + 1, j:j + 2 * pad_col + 1]
			flt_img[i, j] = np.sum(cur * mask)
	return flt_img

def norm_255(img):
	max_val = np.max(img)
	min_val = np.min(img)
	norm_img = ((img - min_val) * 255) / (max_val - min_val)
	return norm_img

smp2 = cv2.imread("hw3_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE)
F = smp2.copy().astype(np.float64)

H1 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 36
H2 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 12
H3 = np.array([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]]) / 12
H4 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 12
H5 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
H6 = np.array([[-1, 2,-1], [0, 0, 0], [1, -2, 1]]) / 4
H7 = np.array([[-1, -2, -1], [2, 4, 2], [-1, -2, -1]]) / 12
H8 = np.array([[-1, 0, 1], [2, 0, -2], [-1, 0, 1]]) / 4
H9 = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 4

# convolution
M1 = conv(F, H1)
M2 = conv(F, H2)
M3 = conv(F, H3)
M4 = conv(F, H4)
M5 = conv(F, H5)
M6 = conv(F, H6)
M7 = conv(F, H7)
M8 = conv(F, H8)
M9 = conv(F, H9)

# energy computation
win_sz = 13
W = np.ones((win_sz, win_sz))
T1 = conv(M1 * M1, W)
T2 = conv(M2 * M2, W)
T3 = conv(M3 * M3, W)
T4 = conv(M4 * M4, W)
T5 = conv(M5 * M5, W)
T6 = conv(M6 * M6, W)
T7 = conv(M7 * M7, W)
T8 = conv(M8 * M8, W)
T9 = conv(M9 * M9, W)

T1 = norm_255(T1)
T2 = norm_255(T2)
T3 = norm_255(T3)
T4 = norm_255(T4)
T5 = norm_255(T5)
T6 = norm_255(T6)
T7 = norm_255(T7)
T8 = norm_255(T8)
T9 = norm_255(T9)

local_features = np.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9])

fig, axs = plt.subplots(3, 3, figsize=(9, 6))

titles = ['Laws 1', 'Laws 2', 'Laws 3', 'Laws 4', 'Laws 5', 'Laws 6', 'Laws 7', 'Laws 8', 'Laws 9']

axs = axs.flatten()

for i in range(9):
    axs[i].imshow(local_features[i, :, :], cmap='gray')
    axs[i].set_title(titles[i])
    axs[i].axis('off')

plt.tight_layout()
plt.show()

# Problem 2. (b)
def kmean(X, k):
	np.random.seed(42)
	cent = X[np.random.choice(X.shape[0], k, replace=False)]
	for i in range(10):
		dist = np.sqrt(((X[:, np.newaxis, :] - cent) ** 2).sum(axis=2))
		lab = np.argmin(dist, axis=1)
		new_cent = np.array([X[lab == j].mean(axis=0) for j in range(k)])
		cent = new_cent
	return lab

local_features = np.moveaxis(local_features, 0, -1)
feature_vectors = local_features.reshape(-1, local_features.shape[2])
k = 4
lab6 = kmean(feature_vectors, k)
clr = np.random.randint(0, 255, (k, 3))
segmented_image = clr[lab6].reshape(local_features.shape[0], local_features.shape[1], 3).astype(np.uint8)
cv2.imwrite("result4.png", segmented_image)

# Problem 2. (c)
tex = [cv2.imread(f"hw3_sample_images/texture{i}.jpg") for i in range(k)]
tex = [cv2.resize(t, (local_features.shape[1], local_features.shape[0])) for t in tex]
res5 = np.zeros((local_features.shape[0], local_features.shape[1], 3), dtype=np.uint8)
for i in range(k):
	mask7 = (lab6.reshape(local_features.shape[0], local_features.shape[1]) == i)
	for c in range(3):
		res5[:, :, c][mask7] = tex[i][:, :, c][mask7]
cv2.imwrite("result5.png", res5)

# Problem 2. (d)
def quilt(img, patch_size, grid_size):
	rows, cols = img.shape
	rows_p, cols_p = patch_size
	patch_collect = []
	for i in range(0, rows - rows_p, rows_p):
		for j in range(0, cols - cols_p, cols_p):
			patch = img[i:i+rows_p, j:j+cols_p]
			patch_collect.append(patch)

	rows_g, cols_g = grid_size
	new_rows, new_cols = rows_p * rows_g, cols_p * cols_g
	res = np.zeros((new_rows, new_cols), dtype=np.uint8)

	for i in range(rows_g):
		for j in range(cols_g):
			patch = patch_collect[np.random.randint(len(patch_collect))]
			res[i*rows_p:(i+1)*rows_p, j*cols_p:(j+1)*cols_p] = patch
	return res


smp3 = cv2.imread("hw3_sample_images/sample3.png", cv2.IMREAD_GRAYSCALE)
patch_size = (300, 300)
grid_size = (20, 20)
res6 = quilt(smp3, patch_size, grid_size)
cv2.imwrite("result6.png", res6)