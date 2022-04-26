How to use the python model:
(last two arguments are non-compulsive)

python predict.py "./test_images/ownDalia.jpg" "./best_model.h5" --category_names="label_map.json" --top_K=5

in case of issues, run:
pip --no-cache-dir install tfds-nightly --user
pip --no-cache-dir install --upgrade tensorflow --user
