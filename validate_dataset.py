
from pathlib import Path
from collections import Counter
import yaml

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "construction-ppe"

with open(DATASET / "data.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

names = config["names"]
if isinstance(names, dict):
    names = [names[i] for i in range(len(names))]

class_counts = Counter()
errors = []
empty_labels = 0

for split in ["train", "val"]:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    labels = {p.stem: p for p in label_dir.glob("*.txt")}

    image_stems = {p.stem for p in images}

    missing = image_stems - labels.keys()
    extra = labels.keys() - image_stems

    print(f"\n========== {split.upper()} ==========")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Missing labels:", len(missing))
    print("Extra labels:", len(extra))

    for stem in missing:
        errors.append(f"{split}: Missing label for {stem}")

    for stem in extra:
        errors.append(f"{split}: No matching image for {stem}")

    for stem in image_stems & labels.keys():
        label_file = labels[stem]

        try:
            lines = label_file.read_text().splitlines()

            if not lines:
                empty_labels += 1
                continue

            for line_no, line in enumerate(lines, 1):
                values = line.split()

                if len(values) != 5:
                    raise ValueError(
                        f"Expected 5 values, got {len(values)}"
                    )

                class_id = int(values[0])
                x, y, w, h = map(float, values[1:])

                if not 0 <= class_id < len(names):
                    raise ValueError(
                        f"Invalid class ID: {class_id}"
                    )

                if not all(0 <= v <= 1 for v in [x, y, w, h]):
                    raise ValueError("Coordinates outside [0, 1]")

                if w <= 0 or h <= 0:
                    raise ValueError("Invalid bounding box dimensions")

                class_counts[class_id] += 1

        except Exception as e:
            errors.append(
                f"{split}/{stem}.txt, line {line_no if 'line_no' in locals() else '?'}: {e}"
            )

print("\n========== CLASS DISTRIBUTION ==========")

for class_id, name in enumerate(names):
    print(f"{class_id}: {name}: {class_counts[class_id]} objects")

print("\nEmpty annotation files:", empty_labels)

print("\n========== VALIDATION SUMMARY ==========")

if errors:
    print("VALIDATION FAILED")
    for error in errors[:30]:
        print("ERROR:", error)
    print("Total errors:", len(errors))
else:
    print("ALL STRUCTURAL AND ANNOTATION CHECKS PASSED!")
