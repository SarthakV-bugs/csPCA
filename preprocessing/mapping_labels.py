import os
import glob


##Mapping the labels
def build_label_map(labels_root):
    label_map = {}
    priorities = [
        "csPCa_lesion_delineations/human_expert/Pooch25",
        "csPCa_lesion_delineations/human_expert/resampled",
        "csPCa_lesion_delineations/human_expert/original",
        "csPCa_lesion_delineations/AI"
    ]

    for folder in priorities:
        path = os.path.join(labels_root, folder)
        for f in glob.glob(os.path.join(path, "*.nii.gz")):
            case_id = os.path.basename(f).split(".")[0]
            if case_id not in label_map:
                label_map[case_id] = f
    return label_map





##Matching stage to match the raw images with the labels

dataset = [] # Build dataset by recursively searching T2W images
def matching_raw_labels(mri_root, label_map):

    for fold in range(5):
        fold_dir = os.path.join(mri_root, f"fold{fold}", f"picai_public_images_fold{fold}")

        # Recursively find all T2W images in this fold
        t2w_files = glob.glob(os.path.join(fold_dir, "**", "*t2w.mha"), recursive=True)

        for t2w_path in t2w_files:
            case_id = os.path.basename(t2w_path).split("_t2w")[0]  # Extract case ID from filename
            if case_id in label_map:
                dataset.append((t2w_path, label_map[case_id]))
            else:
                print(f"No label found for case {case_id}")
    # print(dataset)
    print("Final dataset size (t2w only):", len(dataset))

    return dataset



def main():
    ##Creating a label map
    labels_root = "/home/ibab/PycharmProjects/csPCA/raw_csPCa_data/picai_labels"
    label_map = build_label_map(labels_root)
    print(f"Total labels found: {len(label_map)}")  # should be ~1500


    ##Macthing the raw data with the labelled map
    mri_root = "/home/ibab/PycharmProjects/csPCA/raw_csPCa_data/mri_images"
    matching_dict = matching_raw_labels(mri_root, label_map)
    return matching_dict
    # print(matching_dict)



if __name__ == '__main__':
    data = main()