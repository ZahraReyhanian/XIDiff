import os
import json

def collect_pairs_from_split(split_dir):
    all_pairs = []

    for person in os.listdir(split_dir):
        person_path = os.path.join(split_dir, person)
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

            for n_img in neutral_imgs:
                for e_img in emotion_imgs:
                    all_pairs.append([n_img, e_img, emotion])

    return all_pairs


with open('config/config.json') as f:
    cfg = json.load(f)

root = cfg['root']
dataset_root = os.path.join(root, cfg["dataset_path"])

# collect and save
splits = ['train', 'valid', 'test']
json_path = cfg['json_path']
os.makedirs(json_path, exist_ok=True)

for split in splits:
    split_path = os.path.join(dataset_root, split)
    pairs = collect_pairs_from_split(split_path)

    with open(f'{json_path}/{split}.json', 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"{split}: {len(pairs)} pairs saved.")
