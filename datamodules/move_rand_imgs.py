import os
import shutil
import random

def move_random_images(src_folder, dst_folder_valid, dst_folder_test, percent=0.5):
    if not os.path.exists(src_folder):
        print("src dir does not exit")
        return

    if not os.path.exists(dst_folder_valid):
        print("dst dir does not exit")
        return

    if not os.path.exists(dst_folder_test):
        print("dst dir does not exit")
        return

    image_extensions = ['.jpg']

    for person in os.listdir(src_folder):
        person_path = os.path.join(src_folder, person)
        for emotion in os.listdir(person_path):
            emotion_dir = os.path.join(person_path, emotion)
            dst_emotion_dir_valid = os.path.join(dst_folder_valid, emotion)
            dst_emotion_dir_test = os.path.join(dst_folder_test, emotion)

            all_files = os.listdir(emotion_dir)
            image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]


            num_to_move = int(len(image_files) * percent / 100)
            if num_to_move == 0:
                print("There is no file to transfer")
                return

            # random select for valid
            files_to_move = random.sample(image_files, num_to_move*2)

            # Transfer files
            for file_name in files_to_move[:num_to_move]:
                src_path = os.path.join(emotion_dir, file_name)
                dst_path = os.path.join(dst_emotion_dir_valid, "img_"+file_name)
                shutil.move(src_path, dst_path)
                print(f"Transfer: {file_name}")

            # Transfer files
            for file_name in files_to_move[num_to_move:]:
                src_path = os.path.join(emotion_dir, file_name)
                dst_path = os.path.join(dst_emotion_dir_test, "img_" + file_name)
                print(src_path, dst_path)
                shutil.move(src_path, dst_path)
                print(f"Transfer: {file_name}")

            print(f"{num_to_move*2} فایل با موفقیت منتقل شد.")

move_random_images("E:/uni/Articles/data/MH-FED/train", "E:/uni/Articles/data/MH-FED/valid/All_persons", "E:/uni/Articles/data/MH-FED/test/All_persons")
