import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển đổi sang không gian màu RGB
image = cv2.imread("a01.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Chuyển đổi ảnh thành một mảng 2D (điểm ảnh)
pixels = image_rgb.reshape((-1, 3))

# Sử dụng K-means với k=2
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(pixels)

# Gán nhãn cụm cho từng điểm ảnh
segmented_img = kmeans.labels_.reshape(image_rgb.shape[:2])

# Tạo một ảnh mới cho mỗi cụm màu
clustered_image = np.zeros_like(image_rgb)
for i in range(2):
    clustered_image[segmented_img == i] = kmeans.cluster_centers_[i]

# Hiển thị kết quả
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Ảnh gốc")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Ảnh sau khi phân cụm")
plt.imshow(clustered_image.astype(int))
plt.axis("off")

plt.show()
