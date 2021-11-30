import cv2
import matplotlib.pyplot as plt
import paddlehub as hub
import numpy as np
import math
import matplotlib.image as mpimg
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Some parameter')

    parser.add_argument(
        '--img_path',
        dest='img_path',
        help='The path of the input image',
        type=str,
        default='SS.jpg')
    parser.add_argument(
        '--width',
        dest='width',
        help='The eyebrow width',
        type=int,
        default=17)
    parser.add_argument(
        '--res_img_path',
        dest='res_img_path',
        help='The path of the output image',
        type=str,
        default='res_GT.jpg')
    parser.add_argument(
        '--fat_nums',
        dest='fat_nums',
        help='The number of fat face',
        type=int,
        default=4)
    return parser.parse_args()


# 生成蜡笔小新版眉毛
def thick_eyebrows(image, face_landmark, width):
	for i in range(18-1, 22-1):
		cv2.line(image, face_landmark[i], face_landmark[i+1], (0, 0, 0), width)
	for i in range(23-1, 27-1):
		cv2.line(image, face_landmark[i], face_landmark[i+1], (0, 0, 0), width)
	return image


# 进行胖脸操作
def fat_face(image, face_landmark):
    end_point = face_landmark[30]

    # 胖左脸，3号点到5号点的距离作为胖脸距离
    dist_left = np.linalg.norm(face_landmark[3] - face_landmark[5])
    image = local_traslation_warp(image, face_landmark[3], end_point, dist_left)

    # 胖右脸，13号点到15号点的距离作为胖脸距离
    dist_right = np.linalg.norm(face_landmark[13] - face_landmark[15])
    image = local_traslation_warp(image, face_landmark[13], end_point, dist_right)
    return image


# 局部平移算法
def local_traslation_warp(image, start_point, end_point, radius):
	radius_square = math.pow(radius, 2)
	image_cp = image.copy()

	dist_se = math.pow(np.linalg.norm(end_point - start_point), 2)
	height, width, channel = image.shape
	for i in range(width):
		for j in range(height):
			# 计算该点是否在形变圆的范围之内
			# 优化，第一步，直接判断是会在（start_point[0], start_point[1])的矩阵框中
			if math.fabs(i - start_point[0]) > radius and math.fabs(j - start_point[1]) > radius:
				continue

			distance = (i - start_point[0]) * (i - start_point[0]) + (j - start_point[1]) * (j - start_point[1])

			if distance < radius_square:
				# 计算出（i,j）坐标的原坐标
				# 计算公式中右边平方号里的部分
				ratio = (radius_square - distance) / (radius_square - distance + dist_se)
				ratio = ratio * ratio

				# 映射原位置
				new_x = i + ratio * (end_point[0] - start_point[0])
				new_y = j + ratio * (end_point[1] - start_point[1])

				new_x = new_x if new_x >= 0 else 0
				new_x = new_x if new_x < height - 1 else height - 2
				new_y = new_y if new_y >= 0 else 0
				new_y = new_y if new_y < width - 1 else width - 2

				# 根据双线性插值法得到new_x, new_y的值
				image_cp[j, i] = bilinear_insert(image, new_x, new_y)

	return image_cp


# 双线性插值法
def bilinear_insert(image, new_x, new_y):
	w, h, c = image.shape
	if c == 3:
		x1 = int(new_x)
		x2 = x1 + 1
		y1 = int(new_y)
		y2 = y1 + 1

		part1 = image[y1, x1].astype(np.float) * (float(x2) - new_x) * (float(y2) - new_y)
		part2 = image[y1, x2].astype(np.float) * (new_x - float(x1)) * (float(y2) - new_y)
		part3 = image[y2, x1].astype(np.float) * (float(x2) - new_x) * (new_y - float(y1))
		part4 = image[y2, x2].astype(np.float) * (new_x - float(x1)) * (new_y - float(y1))

		insertvalue = part1 + part2 + part3 + part4

		return insertvalue.astype(np.int8)


def main(args):
	# 加载模型进行人脸关键点检测
	module = hub.Module(name="face_landmark_localization")
	result = module.keypoint_detection(images=[cv2.imread(args.img_path)])

	# 提取出人脸关键点坐标
	face_landmark = np.array(result[0]['data'][0], dtype='int')

	# 生成蜡笔小新版眉毛
	src_img = cv2.imread(args.img_path)
	src_img = thick_eyebrows(src_img, face_landmark, args.width)

	# 进行胖脸操作
	for i in range(1, args.fat_nums):
		src_img = fat_face(src_img, face_landmark)

	cv2.imwrite(args.res_img_path, src_img)

	img1 = mpimg.imread(args.img_path)
	img2 = mpimg.imread(args.res_img_path)
	plt.subplot(1, 2, 1)
	plt.axis('off')
	plt.imshow(img1)
	plt.subplot(1, 2, 2)
	plt.axis('off')
	plt.imshow(img2)
	plt.show()
	print('done')


if __name__ == '__main__':
	args = parse_args()
	main(args)
