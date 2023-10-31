# Prepare

```bash
pip install wandb
git clone https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization.git
cd LIVE-Layerwise-Image-Vectorization
conda create -n live python=3.7
conda activate live
conda install -y pytorch torchvision -c pytorch
conda install -y numpy scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils numba torch-tools scikit-fmm easydict visdom
pip install opencv-python==4.5.4.60  # please install this version to avoid segmentation fault.

cd DiffVG
git submodule update --init --recursive
python setup.py install
cd ..
```

# Prepare the data
```
python3 generate_superpixel-masks.py --data_path=$path_to_imagenet_train
```

# Train
```
wandb login

python -m torch.distributed.launch --nproc_per_node=8  
main_pretrain_svg_superpixel.py --warmup=0 --label_name=128_paths-mask_loss 
--mask_loss --batch_size=64 --wandb --num_workers=4 --data_path=$path_to_imagenet
```