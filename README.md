<p align="center">
  <h1 align="center">ProtoRes</h1>

  <p align="center">
    Proto-Residual Network for Pose Authoring via Learned Inverse Kinematics
    <br>
  </p>
</p>

![Alt Text](./fig/kung_fusupercut.gif)

## Create workspace and clone this repository

```mkdir workspace```

```cd workspace```

```git clone git@github.com:boreshkinai/protores.git```


## Setup : Docker

```
docker build -f Dockerfile -t protores:$USER .

nvidia-docker run -p 18888:8888 -p 16006:6006 -v ~/workspace/protores:/workspace/protores -t -d --shm-size="1g" --name protores_$USER protores:$USER
```
go inside docker container
```
docker exec -i -t protores_$USER  /bin/bash 
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
