for N in {1..2}; do
  python3 detector/align_dataset_mtcnn.py \
    $PWD/data/dataset/raw \
    $PWD/data/dataset/cropped \
  --image_size 160 \
  --margin 32 \
  --random_order \
  --gpu_memory_fraction 0.25 
done 

echo "Splitting the dataset"
python3 split_dataset.py

