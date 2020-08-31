model="Lenet"
#model="Cifar10"
#model="DeepID"
#model="Test"
#model="Caffenet"
#model="Overfeat"
#model="VGG16"

python3 PP_sim.py $model HIR Non-pipeline 1 1
#python3 PP_sim.py $model HIR Pipeline 1 1
#python3 PP_sim.py $model LIR Non-pipeline 1 1
#python3 PP_sim.py $model LIR Pipeline 1 1

#python3 PP_sim.py $model HIR Non-pipeline 2 2
#python3 PP_sim.py $model HIR Pipeline 2 2
#python3 PP_sim.py $model LIR Non-pipeline 2 2
#python3 PP_sim.py $model LIR Pipeline 2 2

#python3 PP_sim.py $model HIR Non-pipeline 4 4
#python3 PP_sim.py $model HIR Pipeline 4 4
#python3 PP_sim.py $model LIR Non-pipeline 4 4
#python3 PP_sim.py $model LIR Pipeline 4 4


