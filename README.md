# nerf-pytorch

![fern sample video](samples/fern.mp4)
![snacks sample video](samples/snacks.mp4)

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