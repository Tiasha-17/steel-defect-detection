# Raw data

The NEU-CLS steel surface defect images (~29 MB, 1,800 images) are not
committed to this repository.

## Download

Source: [NEU Surface Defect Database (NEU-CLS)](https://figshare.com/articles/dataset/NEU-CLS/28903550),
CC BY 4.0, hosted on Figshare -- no account or API token required.

```bash
curl -L -o NEU-CLS.zip https://ndownloader.figshare.com/files/54094775
unzip NEU-CLS.zip -d _extracted
```

The zip extracts in a detection-style layout (`train/train/images/`,
`valid/valid/images/`, plus YOLO-format bounding-box `.txt` labels this
project doesn't need). Pool every image into a folder named after its
class, using the filename prefix (e.g. `scratches_12.jpg` -> `scratches/`):

```bash
for cls in crazing inclusion patches pitted_surface rolled-in_scale scratches; do
  mkdir -p "$cls"
done
find _extracted -iname "*.jpg" | while read -r f; do
  base=$(basename "$f")
  cls=$(echo "$base" | sed -E 's/_[0-9]+\.jpg$//')
  cp "$f" "$cls/$base"
done
rm -rf _extracted NEU-CLS.zip
```

Also mirrored on [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
if you'd rather use `kaggle datasets download` (needs a free account + API token).

## Expected layout

```
data/raw/
├── crazing/*.jpg          # 300 images
├── inclusion/*.jpg        # 300 images
├── patches/*.jpg          # 300 images
├── pitted_surface/*.jpg   # 300 images
├── rolled-in_scale/*.jpg  # 300 images
└── scratches/*.jpg        # 300 images
```

Then, from the repo root:

```bash
python src/prepare_data.py   # writes data/processed/manifest.csv (70/15/15 stratified split)
python src/train_model.py    # trains the classifier head, saves models/defect_classifier.pt + metrics.json
```
