import os
import json
import random

def get_random_target_emotion_file(folder_path):
    """
    Selects a random JPG file from the specified folder.

    Args:
        folder_path (str): The path to the folder containing JPG files.

    Returns:
        str: The full path to a randomly selected JPG file, or None if no JPG files are found.
    """
    jpg_files = []
    try:
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)
            if os.path.isfile(full_path) and entry.lower().endswith(".jpg"):
                jpg_files.append(full_path)

        if jpg_files:
            return random.choice(jpg_files)
        else:
            print(f"No JPG files found in '{folder_path}'")
            return None
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def collect_pairs_from_split(root, split):
    dataset_path = os.path.join(root, split)
    all_pairs = []
    if split == 'train':
        num_neutral_imgs = 2
    else:
        num_neutral_imgs = 10

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        neutral_dir = os.path.join(person_path, 'Neutral')
        if not os.path.exists(neutral_dir):
            continue

        neutral_imgs = [
            os.path.join(neutral_dir, f) for f in os.listdir(neutral_dir)
            if f.endswith('.jpg')
        ]

        for emotion in os.listdir(person_path):
            if emotion == 'Neutral':
                continue

            emotion_dir = os.path.join(person_path, emotion)
            if not os.path.isdir(emotion_dir):
                continue

            emotion_imgs = [
                os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir)
                if f.endswith('.jpg')
            ]

            for n_img in neutral_imgs[:num_neutral_imgs]:
                for e_img in emotion_imgs:
                    if split == 'train':
                        all_pairs.append([n_img, e_img, emotion])
                    else:
                        id_name = os.path.basename(n_img).replace('.jpg', '')
                        train_path = os.path.join(root, 'train')
                        id_path = os.path.join(train_path, id_name)
                        id_exp_dir = os.path.join(str(id_path), str(emotion))
                        target = get_random_target_emotion_file(id_exp_dir)

                        all_pairs.append([n_img, e_img, emotion, target])

    return all_pairs


with open('../config/config.json') as f:
    cfg = json.load(f)

root = str(cfg["root"])
dataset_root = os.path.join(root, cfg["dataset_path"])

# collect and save
splits = ['train', 'valid', 'test']
json_path = os.path.join(root, cfg['json_path'])
print('saved path: ', json_path)
os.makedirs(json_path, exist_ok=True)

for split in splits:
    pairs = collect_pairs_from_split(root=dataset_root, split=split)

    with open(f'{json_path}/{split}.json', 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"{split}: {len(pairs)} pairs saved.")
