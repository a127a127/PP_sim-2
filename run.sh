#model="Lenet"
#model="Cifar10"
#model="DeepID"
model="Caffenet"
#model="Overfeat"
#model="Test"

# model="VGG16" 

python3 PP_sim.py $model LIDR Non-pipeline 1 1 64
python3 PP_sim.py $model LIDR Pipeline 1 1 64
python3 PP_sim.py $model HIDR Non-pipeline 1 1 64
python3 PP_sim.py $model HIDR Pipeline 1 1 64

python3 PP_sim.py $model LIDR Non-pipeline 2 2 64
python3 PP_sim.py $model LIDR Pipeline 2 2 64
python3 PP_sim.py $model HIDR Non-pipeline 2 2 64
python3 PP_sim.py $model HIDR Pipeline 2 2 64

python3 PP_sim.py $model LIDR Non-pipeline 4 4 64
python3 PP_sim.py $model LIDR Pipeline 4 4 64
python3 PP_sim.py $model HIDR Non-pipeline 4 4 64
python3 PP_sim.py $model HIDR Pipeline 4 4 64

