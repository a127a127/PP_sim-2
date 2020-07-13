#model="Caffenet"
model="Overfeat"
#model="VGG16"

python3 PP_sim.py $model SCF Non_pipeline
#python3 PP_sim.py $model SCF Pipeline
#python3 PP_sim.py $model SRF Non_pipeline
#python3 PP_sim.py $model SRF Pipeline

#python3 PP_sim.py $model SCFParal Non_pipeline 2
#python3 PP_sim.py $model SCFParal Pipeline 2
#python3 PP_sim.py $model SRFParal Non_pipeline 2
#python3 PP_sim.py $model SRFParal Pipeline 2

#python3 PP_sim.py $model SCFParal Non_pipeline 4
#python3 PP_sim.py $model SCFParal Pipeline 4
#python3 PP_sim.py $model SRFParal Non_pipeline 4
#python3 PP_sim.py $model SRFParal Pipeline 4

