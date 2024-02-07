import csv
import json
from pathlib import Path
import argparse
from colorama import Fore

# Assigns class labels to ADNI images based on the ADNI csv file
# writes/updates json file containing all the images fed into training script: Adni2dDiffusionAnomalyDetection.py
# (metadata file that comes with ADNI dataset from https://adni.loni.usf.edu/)

# 1: CN, 2: EMCI, 3: MCI, 4: LMCI, 5: AD
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcJsonFile', type=str, required=True, help='json file passed as input to Adni2dDiffusionAnomalyDetection.py --training')
    parser.add_argument('--adniCsvMapFile', type=str, required=True, help='csv file thats comes along with ADNI dataset')
    parser.add_argument('--outFile', type=str, required=True)
    args = parser.parse_args()

    with open(args.srcJsonFile, 'r') as f:
        json_data = json.load(f)

    image_to_class = {}
    with open(args.adniCsvMapFile) as f:
        csvreader = csv.reader(f, delimiter=',')
        for row in csvreader:
            if row[2] == 'CN':
                image_to_class[row[0]] = 1
            elif row[2] == 'EMCI':
                image_to_class[row[0]] = 2
            elif row[2] == 'MCI':
                image_to_class[row[0]] = 3
            elif row[2] == 'LMCI':
                image_to_class[row[0]] = 4
            elif row[2] == 'AD':
                image_to_class[row[0]] = 5

    # print(f'images_to_class: {image_to_class}')

    for image_d in json_data:
        image_path = Path(image_d['image'])
        image_fname = image_path.stem.split('.')[0]
        image_id = image_fname.split('_')[-1]
        # print(f'image_id: {image_id}')
        if image_id not in image_to_class:
            print(f'{image_id} not in image_to_class')
            continue
        image_d['slice_label'] = image_to_class[image_id]

    with open(args.outFile, 'w') as f:
        json.dump(json_data, f)

# print out class mapping 
class_mapping = {
  'CN': 1,
  'EMCI': 2,
  'MCI': 3,
  'LMCI': 4,
  'AD': 5
}
print("\n-------------------")
for k, v in class_mapping.items():
    print(f'{Fore.BLUE}{k}\t:{Fore.MAGENTA}{v}{Fore.RESET}')
print("-------------------")
