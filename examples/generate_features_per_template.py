import os
import json
import csv
from ast import literal_eval
import calculate_metrics
from torch_utils import distributed as dist



def build_label_json(root_dir, output_file):
    label_dict = {}

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                rel_path = os.path.relpath(os.path.join(subdir, file), root_dir)
                rel_path = rel_path.replace("\\", "/")  # for Windows compatibility
                label_dict[rel_path] = [int(x) for x in os.path.splitext(rel_path)[0].split('/')]

    output_data = {"labels": label_dict}

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
if __name__ == "__main__":
    # Initialize dataset.json
    image_dir = "data/templates_per_classid"  # replace with your actual image root
    json_file = os.path.join(image_dir, "dataset.json")
    build_label_json(image_dir, json_file)

    # Save features
    feature_dir  = f"data/features_per_classid"
    os.makedirs(feature_dir, exist_ok=True)
    calculate_metrics.generate_reference_stats(
        image_path      = image_dir,
        dest_path       = None,
        metrics         = ['fid', 'fd_dinov2'],
        max_batch_size  = 256,
        num_workers     = 2,
        feature_dir     = feature_dir,
        use_labels      = True,     # Expose labels from dataset.json to StatsIterable
    )

    if dist.get_rank() == 0:
        calculate_metrics.merge_metric_feature_directories(feature_dir)
        calculate_metrics.merge_feature_csvs(feature_dir)

        # Write it to the right format [classid, exampple] -> class_id, example
        with open(os.path.join(feature_dir, "features.csv") , "r", newline="") as f:
            reader = csv.reader(f)
            rows = [[int(x) for x in literal_eval(row[0])] for row in reader]

        with open(os.path.join(feature_dir, "features.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


