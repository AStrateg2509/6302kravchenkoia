import csv
import random
import requests
import json
import os
from pathlib import Path


def get_paintings(csv_path):
    painting_objects = []

    with open(csv_path, 'r', encoding='utf-8-sig') as csv_path:
        reader = csv.DictReader(f=csv_path)
        for row in reader:
            if row.get("Classification") == "Paintings":
                print(row)
                object_number = row.get('Object Number')
                if object_number:
                    painting_objects.append({
                        'object_number': object_number,
                        'object_id': row.get('Object ID')
                    })

        return painting_objects


def download_random_painting(csv_path, output_dir='paintings'):
    paints = get_paintings(csv_path)
    image_url = 0
    while not image_url:
        object_id = random.choice(paints)["object_id"]
        print(object_id)
        url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
        response = requests.get(url)
        data = response.json()
        image_url = data.get("primaryImage")
        print(response, data, image_url)

    os.makedirs(output_dir, exist_ok=True)

    image_response = requests.get(image_url)

    image_path = os.path.join(output_dir, f"{object_id}.jpg")
    json_path = os.path.join(output_dir, f"{object_id}.json")

    with open(image_path, "wb") as img_file:
        img_file.write(image_response.content)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f"Сохранено:\n{image_path}\n{json_path}")


def main():
    csv_path = 'MetObjects.csv'

    if not os.path.exists(csv_path):
        print(f"Файл {csv_path} не найден!")

    download_random_painting(csv_path=csv_path)


if __name__ == "__main__":
    main()
