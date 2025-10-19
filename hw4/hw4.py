import numpy as np
import cv2

# Problem 1. (a)
smp1 = cv2.imread("hw4_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)

I2 = np.array([[1, 2], [3, 0]])
N1 = len(I2)

thr1 = 255 * (I2 + 0.5) / (N1 * N1)
rows1, cols1 = smp1.shape
thrm1 = np.tile(thr1, (rows1 // N1, cols1 // N1))

res1 = np.where(smp1 > thrm1, 255, 0).astype(np.uint8)

cv2.imwrite("result1.png", res1)

# Problem 1. (b)
I = np.array([[1, 2], [3, 0]])
while I.shape[0] < 256:
    I = np.block([[4 * I + 1, 4 * I + 2], [4 * I + 3, 4 * I]])

I256 = I
N2 = len(I256)
thr2 = 255 * (I256 + 0.5) / (N2 * N2)
rows2, cols2 = smp1.shape
thrm2 = np.tile(thr2, (rows2 // N2, cols2 // N2))

res2 = np.where(smp1 > thrm2, 255, 0).astype(np.uint8)

cv2.imwrite("result2.png", res2)

# Problem 1. (c)
# Floyd-Steinberg
def floyd(img):
    img = img.astype(np.float32)
    rows, cols = img.shape
    mask = np.array([[0, 0, 7 / 16], [3 / 16, 5 / 16, 1 / 16]], dtype=np.float32)

    for y in range(rows):
        for x in range(cols):
            old = img[y, x]
            new = 0 if old < 128 else 255
            error = old - new
            img[y, x] = new

            for ky in range(2):
                for kx in range(-1, 2):
                    ny = y + ky
                    nx = x + kx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        img[ny, nx] += error * mask[ky, kx + 1]
    return np.clip(img, 0, 255).astype(np.uint8)

# Jarvis’
def jarvis(img):
    img = img.astype(np.float32)
    rows, cols = img.shape
    mask = np.array([[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]]) / 48.0

    for y in range(rows):
        for x in range(cols):
            old = img[y, x]
            new = 0 if old < 128 else 255
            error = old - new
            img[y, x] = new

            for ky in range(3):
                for kx in range(-2, 3):
                    ny = y + ky
                    nx = x + kx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        img[ny, nx] += error * mask[ky, kx + 2]
    return np.clip(img, 0, 255).astype(np.uint8)

res3 = floyd(smp1)
cv2.imwrite("result3.png", res3)

res4 = jarvis(smp1)
cv2.imwrite("result4.png", res4)

# Problem 2.
def flood_fill2(img, labs, i, j, lab):
    rows, cols = img.shape
    stack = [(i, j)]
    while stack:
        ci, cj = stack.pop()
        if 0 <= ci < rows and 0 <= cj < cols and labs[ci, cj] == 0 and img[ci, cj] == 0:
            labs[ci, cj] = lab
            stack.append((ci - 1, cj))
            stack.append((ci + 1, cj))
            stack.append((ci, cj - 1))
            stack.append((ci, cj + 1))

def cnt_comp(img):
    rows, cols = img.shape
    labs = np.zeros_like(img, dtype=np.uint32)
    lab = 1
    for i in range(rows):
        for j in range(cols):
            if img[i, j] == 0 and labs[i, j] == 0:
                flood_fill2(img, labs, i, j, lab)
                lab += 1
    cnt = lab - 1
    return cnt, labs

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

# get bounding box
def bound_box(crd):
    x = crd[:, 0, 0]
    y = crd[:, 0, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1

# analyze objects
def analyze(img, labs, objects_num, cover):
    res = []

    for label in range(1, objects_num + 1):
        cur = (labs == label).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(cur, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        hierarchy = hierarchy[0] if hierarchy is not None else []
        holes = sum(1 for h in hierarchy if h[3] != -1)
        main_contour = max(contours, key=cv2.contourArea)
        x_min, y_min, w, h = bound_box(main_contour)

        aspect_ratio = h / w if w else 0
        coverage_ratio = np.sum(cur == 255) / (h * w)

        bays = 0
        hull = cv2.convexHull(main_contour, returnPoints=False)
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(main_contour, hull)
            if defects is not None:
                for defect in defects[:, 0]:
                    start, end, farth, _ = defect
                    triangle = np.array([main_contour[i][0] for i in (start, end, farth)])
                    if cv2.contourArea(triangle) > cover * (h * w):
                        bays += 1

        res.append({"label": label, "lakes": holes, "bays": bays, "coverage_ratio": coverage_ratio, "aspect_ratio": aspect_ratio, "min_x" : x_min})
    return res

# extend bottom music staff
def extend_lines(labs_line, line_img, ref, tar):
    rys, rxs = np.where(labs_line == ref)
    rx_min = rxs.min()
    rx_max = rxs.max()

    tys, txs = np.where(labs_line == tar)
    ty = tys[0]

    rows, cols = line_img.shape
    for x in range(rx_min, rx_max + 1):
        if 0 <= x < cols and 0 <= ty < rows:
            line_img[ty][x] = 0
    return

# remove noise line
def remove_short_lines(img, labs, min_len):
    res = img.copy()
    for label in range(1, np.max(labs) + 1):
        xs = np.where(labs == label)[1]
        if xs.size and (xs.max() - xs.min() + 1) < min_len:
            res[labs == label] = 255
    return res

# insert additional lines between the staff lines
def insert_lines(labs_line, line_img, ref):
    rows, cols = labs_line.shape
    label_to_y = {}
    for label in range(1, np.max(labs_line) + 1):
        ys, xs = np.where(labs_line == label)
        cnt = np.bincount(ys)
        max_y = np.argmax(cnt)
        label_to_y[label] = max_y

    sorted_labels = sorted(label_to_y.items(), key=lambda x: x[1])
    sorted_ys = [y for _, y in sorted_labels]
    rys, rxs = np.where(labs_line == ref)
    rx_min = rxs.min()
    rx_max = rxs.max()

    for i in range(len(sorted_ys) - 1):
        y1, y2 = sorted_ys[i], sorted_ys[i + 1]
        mid_y = (y1 + y2) // 2
        if 0 <= mid_y < rows:
            line_img[mid_y, rx_min:rx_max + 1] = 0
    return

# get closest label and its minimum distance
def get_closest_lab_dist(labs, labs_line, tar, labs_line_cnt):
    labs_line_main_y = {} 
    ys, xs = np.where(labs_line > 0)
    labs_line_main_y = {label: np.bincount(ys[labs_line[ys, xs] == label]).argmax() for label in range(1, labs_line_cnt + 1) if np.any(labs_line[ys, xs] == label)}

    tys, txs = np.where(labs == tar)
    labs_y_max = tys.max()
    dist = {label: abs(y - labs_y_max) for label, y in labs_line_main_y.items()}
    closest_label = min(dist, key=dist.get)
    min_dist = dist[closest_label]

    return closest_label, min_dist

# classification
def classify(obj):
    if obj['coverage_ratio'] > 0.9 or obj['aspect_ratio'] > 3.5:
        return 0 # unknown
    elif obj['lakes'] > 1:
        return 1 # treble clef
    elif obj['lakes'] == 1:
        if obj['bays'] == 8:
            return 12 # sharp
        elif obj['bays'] == 2:
            if obj['aspect_ratio'] > 2:
                return 11 # natural
            elif obj['aspect_ratio'] > 1:
                return 5 # 16th notes
            elif obj['aspect_ratio'] < 1:
                return 9 # 2 beamed 16th notes
        elif obj['bays'] == 1:
            if obj['coverage_ratio'] < 0.28:
                return 4 # half note
            else:
                return 10 # flat
    elif obj['lakes'] == 0:
        if obj['aspect_ratio'] < 0.5:
            return 8 # whole or flat rest
        elif obj['bays'] == 4:
            return 13 # quarter rest
        elif obj['bays'] == 1:
            return 2 # quarter note
        elif obj['bays'] == 3:
            return 14 # 16th rest
        elif obj['bays'] == 2:
            if obj['aspect_ratio'] > 2:
                return 15 # 8th rest
            elif obj['aspect_ratio'] > 1:
                return 3 # 8th notes
            elif obj['aspect_ratio'] > 0.95:
                return 6 # 2 beamed 8th notes
            else:
                return 16 # 3 beamed 16th notes
    else:
        return 0 # unknown

smp2 = cv2.imread("hw4_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE)
binary = np.where(smp2 > 175, 255, 0).astype(np.uint8)
mask1 = np.ones((1, 15), dtype=np.uint8)
mask2 = np.ones((3, 1), dtype=np.uint8)
# get staff image
dil = dilate(binary, mask1)
line_img = erode(dil, mask1)

# clean and extend staff image
line_img_ex = line_img.copy()
_, labs_line = cnt_comp(line_img_ex)
extend_lines(labs_line, line_img_ex, 1, np.max(labs_line))
_, labs_line = cnt_comp(line_img_ex)
line_img_ex = remove_short_lines(line_img_ex, labs_line, 100)
_, labs_line = cnt_comp(line_img_ex)

# insertion
insert_lines(labs_line, line_img_ex, 1)
labs_line_cnt, labs_line = cnt_comp(line_img_ex)

# get note image
inv_line_img = 255 - line_img
note_img = binary + inv_line_img
ero = erode(note_img, mask2)
note_img = dilate(ero, mask2)
labs_cnt, labs = cnt_comp(note_img)

objs = analyze(note_img, labs, labs_cnt, 0.05)
scale_map = {11 : 'D', 10 : 'E', 9 : 'F', 8 : 'G', 7 : 'A', 6 : 'B'}
print("sample2.png recognition result:")
print('[', end='')
sorted_objs = sorted(objs, key=lambda x: x['min_x'])
valid_objs = [ob for ob in sorted_objs if classify(ob) != 0]
for i, obj in enumerate(valid_objs):
    tag = classify(obj)
    if tag != 0:
        closest_label, min_dist = get_closest_lab_dist(labs, labs_line, obj['label'], labs_line_cnt) 
        is_last = i == len(valid_objs) - 1
        end_token = '' if is_last else ', '
        if tag == 8:
            if scale_map[closest_label] == 'B':
                print('7', end=end_token)
            else:
                print('8', end=end_token) 
        elif tag not in [2, 3, 4, 5, 6, 9, 16]:
            print(f'{tag}', end=end_token)
        elif min_dist > 3:
            print(f"{tag}(C)", end=end_token)
        else:
            print(f"{tag}({scale_map[closest_label]})", end=end_token)
print(']', end='\n')

smp3 = cv2.imread("hw4_sample_images/sample3.png", cv2.IMREAD_GRAYSCALE)
binary = np.where(smp3 > 175, 255, 0).astype(np.uint8)
mask1 = np.ones((1, 17), dtype=np.uint8)
mask2 = np.ones((3, 1), dtype=np.uint8)
# get staff image
dil = dilate(binary, mask1)
line_img = erode(dil, mask1)

# extend staff image
line_img_ex = line_img.copy()
_, labs_line = cnt_comp(line_img_ex)
extend_lines(labs_line, line_img_ex, 1, np.max(labs_line))

# insertion
insert_lines(labs_line, line_img_ex, 1)
labs_line_cnt, labs_line = cnt_comp(line_img_ex)

# get note image
inv_line_img = 255 - line_img
note_img = binary + inv_line_img
ero = erode(note_img, mask2)
note_img = dilate(ero, mask2)
labs_cnt, labs = cnt_comp(note_img)

objs = analyze(note_img, labs, labs_cnt, 0.02)
scale_map = {11 : 'D', 10 : 'E', 9 : 'F', 8 : 'G', 7 : 'A', 6 : 'B'}
print("sample3.png recognition result:")
print('[', end='')
sorted_objs = sorted(objs, key=lambda x: x['min_x'])
valid_objs = [ob for ob in sorted_objs if classify(ob) != 0]
for i, obj in enumerate(valid_objs):
    tag = classify(obj)
    if tag != 0:
        closest_label, min_dist = get_closest_lab_dist(labs, labs_line, obj['label'], labs_line_cnt) 
        is_last = i == len(valid_objs) - 1
        end_token = '' if is_last else ', '
        if tag == 8:
            if scale_map[closest_label] == 'B':
                print('7', end=end_token)
            else:
                print('8', end=end_token) 
        elif tag not in [2, 3, 4, 5, 6, 9, 16]:
            print(f'{tag}', end=end_token)
        elif min_dist > 3:
            print(f"{tag}(C)", end=end_token)
        else:
            print(f"{tag}({scale_map[closest_label]})", end=end_token)
print(']', end='\n')

# Problem 3.
# get notch centers
def get_notch_centers(img, horizontal, vertical):
    rows, cols = img.shape
    ctr_y, ctr_x = rows // 2, cols // 2
    notch_ctrs = []
    if horizontal:
        for x in range(cols):
            if abs(x - ctr_x) > 30:
                notch_ctrs.append((ctr_y, x))
    if vertical:
        for y in range(rows):
            if abs(y - ctr_y) > 30:
                notch_ctrs.append((y, ctr_x))
    return notch_ctrs

# filter the image
def filter_image(img, notch_ctrs, fshift):
    rows, cols = img.shape
    rad = 10
    mask = np.ones((rows, cols), dtype=np.float32)
    for cy, cx in notch_ctrs:
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - cx) ** 2 + (y - cy) ** 2 <= rad ** 2
        mask[mask_area] = 0
    
    fshift_filtered = fshift * mask

    f_ishift = np.fft.ifftshift(fshift_filtered)
    res = np.fft.ifft2(f_ishift)
    res = np.real(res)

    res = ((res - res.min()) * 255 / (res.max() - res.min())).astype(np.uint8)
    return res

smp4 = cv2.imread("hw4_sample_images/sample4.png", cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(smp4)
fshift = np.fft.fftshift(f)

# for frequency domain
mag_spec = 20 * np.log(np.abs(fshift) + 1)
mag_norm = ((mag_spec - mag_spec.min()) * 255 / (mag_spec.max() - mag_spec.min())).astype(np.uint8)

notch_ctrs_5 = get_notch_centers(smp4, horizontal=False, vertical=True)
res5 = filter_image(smp4, notch_ctrs_5, fshift)
cv2.imwrite("result5.png", res5)

notch_ctrs_6 = get_notch_centers(smp4, horizontal=True, vertical=False)
res6 = filter_image(smp4, notch_ctrs_6, fshift)
cv2.imwrite("result6.png", res6)

notch_ctrs_7 = get_notch_centers(smp4, horizontal=True, vertical=True)
res7 = filter_image(smp4, notch_ctrs_7, fshift)
cv2.imwrite("result7.png", res7)