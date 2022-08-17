<p align="center">
  <h1 align="center">SMPL-IK</h1>

  <p align="center">
    Learned Morphology-Aware Inverse Kinematics for AI Driven Artistic Workflows
    <br>
  </p>
</p>

![Alt Text](./fig/teaser.png)

## Create workspace and clone this repository

```mkdir workspace```

```cd workspace```

```git clone git@github.com:boreshkinai/smpl-ik.git```


## Setup : Docker

```
docker build -f Dockerfile -t smpl-ik:$USER .

nvidia-docker run -p 18888:8888 -p 16006:6006 -v ~/workspace/smpl-ik:/workspace/smpl-ik -t -d --shm-size="1g" --name smpl-ik_$USER smpl-ik:$USER
```
go inside docker container
```
docker exec -i -t smpl-ik_$USER  /bin/bash 
```
launch training session
```
python run.py --config=protores/configs/experiments/encoder_blocks.yaml
```

## Setup : Conda
```
conda create --name protores python=3.8

conda activate protores 

pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
To use notebooks (optional) :
```
conda install jupyter
```

## Citation

If you use ProtRes in any context, please cite the following paper:

```
@inproceedings{oreshkin2022protores:,
  title={ProtoRes: Proto-Residual Network for Pose Authoring via Learned Inverse Kinematics},
  author={Boris N. Oreshkin and Florent Bocquelet and F{\'{e}}lix G. Harvey and Bay Raitt and Dominic Laflamme},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
