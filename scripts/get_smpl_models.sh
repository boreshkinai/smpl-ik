DEFAULT_MODEL_PATH=/model_files

mkdir -p ${1-${DEFAULT_MODEL_PATH}}

cd ${1-${DEFAULT_MODEL_PATH}} 

aria2c -x 10 -j 10  https://storage.googleapis.com/unity-rd-ml-graphics-deeppose/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl
    
aria2c -x 10 -j 10  https://storage.googleapis.com/unity-rd-ml-graphics-deeppose/smpl/basicModel_m_lbs_10_207_0_v1.0.0.pkl
