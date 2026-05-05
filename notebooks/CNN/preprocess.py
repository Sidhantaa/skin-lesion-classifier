import cv2
import os
import numpy as np 
from PIL import Image
import splitfolders

# Stratify split the training set 80/20 into a val set
def create_train_val_split(RAW_TRAIN, SPLIT_OUTPUT):
    splitfolders.ratio(
        input=RAW_TRAIN,
        output=SPLIT_OUTPUT,
        seed=42,
        ratio=(0.8, 0.2),
        move=False)
    
    print("Split complete.")

# Remove hairs using Dull Razor algorithm
def dull_razor(image):
  ker1_size = (9,9)
  ker2_size = (3,3)
  threshold = 10
  maxval = 255
  radius = 6

  grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,ker1_size)
  blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

  bhg = cv2.GaussianBlur(blackhat,ker2_size,cv2.BORDER_DEFAULT)

  ret,mask = cv2.threshold(bhg,threshold,maxval,cv2.THRESH_BINARY)

  dst = cv2.inpaint(image,mask,radius,cv2.INPAINT_TELEA)

  img_clean = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

  return img_clean

# resizes shortest side to 256 and then center crops a 224x224 area
def resize_crop(img, resize_size=256, crop_size=224):
    width, height = img.size

    if width < height:
        new_width = resize_size
        new_height = max(int((resize_size / width) * height), crop_size)   
    else:
        new_height = resize_size
        new_width = max(int((resize_size / height) * width), crop_size)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    width, height = img.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    return img.crop((left, top, right, bottom))

# Function to apply both processes and save color image
def process_image(image_bgr):
    img_rgb = dull_razor(image_bgr)
    img_pil = Image.fromarray(img_rgb)
    img_pil = resize_crop(img_pil)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Apply preprocessing to all images in a dir and save to a new
def process_directory(input_dir, output_dir):

    output_root = output_dir

    print(f"Processing: {input_dir}")
    print(f"Saving to: {output_root}\n")

    for subfolder_path, _, imnames in os.walk(input_dir):

        count = 0

        for imname in imnames:
            if imname.lower().endswith((".jpg")):

                image_path = os.path.join(subfolder_path, imname)
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Failed: {image_path}")
                    continue

                processed = process_image(img)

                relative_path = os.path.relpath(subfolder_path, input_dir)
                save_folder = os.path.join(output_root, relative_path)
                os.makedirs(save_folder, exist_ok=True)

                save_path = os.path.join(save_folder, imname)
                cv2.imwrite(save_path, processed)

                count += 1

        if count > 0:
            print(f"{os.path.basename(subfolder_path)} done ({count})")

    print("\nDone.")

if __name__ == "__main__":

    RAW_TRAIN = "data/Train"
    SPLIT_OUTPUT = "data/split_data"

    # Split data
    create_train_val_split(RAW_TRAIN, SPLIT_OUTPUT)

    # Process directories
    process_directory(input_dir=f"{SPLIT_OUTPUT}/train",output_dir="data/processed/train")
    process_directory(input_dir=f"{SPLIT_OUTPUT}/val",output_dir="data/processed/val")
    process_directory(input_dir="data/Test", output_dir="data/processed/test")