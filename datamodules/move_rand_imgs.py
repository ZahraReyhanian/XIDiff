import os
import shutil
import random

def move_random_images(src_folder, dst_folder, percent=10):
    if not os.path.exists(src_folder):
        print("src dir does not exit")
        return

    if not os.path.exists(dst_folder):
        print("dst dir does not exit")
        return

    image_extensions = ['.jpg']

    for person in os.listdir(src_folder):
        person_path = os.path.join(src_folder, person)
        for emotion in os.listdir(person_path):
            emotion_dir = os.path.join(person_path, emotion)
            dst_emotion_dir = os.path.join(dst_folder, emotion)

            all_files = os.listdir(emotion_dir)
            image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]


            num_to_move = int(len(image_files) * percent / 100)
            if num_to_move == 0:
                print("There is no file to transfer")
                return

            # random select
            files_to_move = random.sample(image_files, num_to_move)

            # Transfer files
            for file_name in files_to_move:
                src_path = os.path.join(emotion_dir, file_name)
                dst_path = os.path.join(dst_emotion_dir, "img_"+file_name)
                shutil.move(src_path, dst_path)
                print(f"Transfer: {file_name}")

            print(f"{num_to_move} فایل با موفقیت منتقل شد.")

# move_random_images("train_path", "valid_or_test_path/All_persons", 10)
