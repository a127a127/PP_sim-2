model="Lenet"
# model="Cifar10"
# model="DeepID"
# model="Caffenet"
# model="Overfeat"
# model="VGG16"
# model="Test"

python3 PP_sim_o.py $model SRF Non-pipeline 1 1
python3 PP_sim_o.py $model SRF Pipeline 1 1
python3 PP_sim_o.py $model SCF Non-pipeline 1 1
python3 PP_sim_o.py $model SCF Pipeline 1 1

python3 PP_sim_o.py $model SRF Non-pipeline 2 2
python3 PP_sim_o.py $model SRF Pipeline 2 2
python3 PP_sim_o.py $model SCF Non-pipeline 2 2
python3 PP_sim_o.py $model SCF Pipeline 2 2

python3 PP_sim_o.py $model SRF Non-pipeline 4 4
python3 PP_sim_o.py $model SRF Pipeline 4 4
python3 PP_sim_o.py $model SCF Non-pipeline 4 4
python3 PP_sim_o.py $model SCF Pipeline 4 4
