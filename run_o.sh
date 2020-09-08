model="Lenet"
#model="Cifar10"
#model="DeepID"
#model="Test"
#model="Caffenet"
#model="Overfeat"
#model="VGG16"

python3 PP_sim_o.py $model SRF Non-pipeline
python3 PP_sim_o.py $model SRF Pipeline
python3 PP_sim_o.py $model SCF Non-pipeline
python3 PP_sim_o.py $model SCF Pipeline

python3 PP_sim_o.py $model SRFParal Non-pipeline 2
python3 PP_sim_o.py $model SRFParal Pipeline 2
python3 PP_sim_o.py $model SCFParal Non-pipeline 2
python3 PP_sim_o.py $model SCFParal Pipeline 2

python3 PP_sim_o.py $model SRFParal Non-pipeline 4
python3 PP_sim_o.py $model SRFParal Pipeline 4
python3 PP_sim_o.py $model SCFParal Non-pipeline 4
python3 PP_sim_o.py $model SCFParal Pipeline 4


