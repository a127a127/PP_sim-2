# non-pipeline
python3 PP_sim.py 0 0 > ./statistics/Default_Mapping/Non_pipeline/result.txt
python3 PP_sim.py 1 0 > ./statistics/High_Parallelism_Mapping/Non_pipeline/result.txt
python3 PP_sim.py 2 0 > ./statistics/Same_Column_First_Mapping/Non_pipeline/result.txt

# pipeline
python3 PP_sim.py 0 1 > ./statistics/Default_Mapping/Pipeline/result.txt
python3 PP_sim.py 1 1 > ./statistics/High_Parallelism_Mapping/Pipeline/result.txt
python3 PP_sim.py 2 1 > ./statistics/Same_Column_First_Mapping/Pipeline/result.txt