# nerf-pytorch

https://user-images.githubusercontent.com/4343568/169726055-6a4445fe-a118-4d4d-b728-2ea866e8070a.mp4

https://user-images.githubusercontent.com/4343568/169726068-cffa00b1-2670-41a1-bde0-38549775c988.mp4

## Usage

First, download and extract LLFF data in the data directory

```bash
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIG19/lffusion/testscene.zip
unzip testscene.zip
```

So that data directories is placed under data directory, like this:

```
data/fern/images_4/*.png
data/fern/pose_bounds.npy
```

images_x corresponds to 1/x downsampled images.

Then you can run train script

```bash
python scripts/train.py --conf configs/llff.py
```

After the training is finished, you can generate sample videos:

```bash
python scripts/generate_video.py --conf configs/llff.py --ckpt [CHECKPOIT PATH] out=samples.mp4
```
