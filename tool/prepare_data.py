### tools/prepare_data.py
import os
import cv2
import numpy as np
import glob


def cv2_imread_safe(file_path):
    """能够读取中文路径的 OpenCV 读取函数"""
    try:
        img_array = np.fromfile(file_path, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[Error] 读取失败: {file_path}, {e}")
        return None


def cv2_imwrite_safe(file_path, img):
    """能够保存中文路径的 OpenCV 保存函数"""
    try:
        is_success, buffer = cv2.imencode(".jpg", img)
        if is_success:
            buffer.tofile(file_path)
            return True
    except Exception as e:
        print(f"[Error] 保存失败: {file_path}, {e}")
    return False


def augment_brightness(img, val):
    """调整亮度 (防止溢出修复版)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 转为 int32 进行运算，防止溢出
    v_int = v.astype(np.int32)
    v_int = v_int + val
    v = np.clip(v_int, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def augment_noise(img, sigma=25):
    """添加高斯噪声"""
    row, col, ch = img.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = img + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def process_pipeline(source_dir, target_dir):
    if not os.path.exists(source_dir):
        print(f"[Error] 源目录 '{source_dir}' 不存在。")
        return

    # 搜索图片
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(source_dir, ext)))
    for ext in extensions:
        files.extend(glob.glob(os.path.join(source_dir, ext.upper())))
    files = sorted(list(set(files)))

    print(f"[Info] 找到 {len(files)} 个文件待处理...")

    count_person = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        if name.startswith('.'): continue

        img = cv2_imread_safe(file_path)
        if img is None: continue

        person_dir = os.path.join(target_dir, name)
        os.makedirs(person_dir, exist_ok=True)

        # === 1. 保存原图 ===
        cv2_imwrite_safe(os.path.join(person_dir, "0.jpg"), img)

        # === 2. 生成扩增图 (仅亮度与噪声) ===
        # 警告：严禁在此处使用平移(Shift)，会导致 PCA 特征对其失效
        variations = []

        # 亮度组
        variations.append(augment_brightness(img, 30))
        variations.append(augment_brightness(img, -30))
        variations.append(augment_brightness(img, 50))
        variations.append(augment_brightness(img, -50))  # 新增

        # 噪声组
        variations.append(augment_noise(img, sigma=10))
        variations.append(augment_noise(img, sigma=20))
        variations.append(augment_noise(img, sigma=30))

        # 混合组 (亮+噪)
        v_bright = augment_brightness(img, 20)
        variations.append(augment_noise(v_bright, sigma=15))

        v_dark = augment_brightness(img, -20)
        variations.append(augment_noise(v_dark, sigma=15))

        for idx, var_img in enumerate(variations):
            cv2_imwrite_safe(os.path.join(person_dir, f"{idx + 1}.jpg"), var_img)

        print(f"   -> 已处理: {name} (1+9张)")
        count_person += 1

    print(f"处理完成。共 {count_person} 人。")


if __name__ == "__main__":
    src = input("请输入原始证件照目录 [默认: raw_photos]: ").strip() or "raw_photos"
    dst = "dataset"
    process_pipeline(src, dst)