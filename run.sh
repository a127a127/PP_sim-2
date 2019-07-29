# non-pipeline
python3 PP_sim.py 0 0 > ./statistics/non_pipeline/DefaultMapping/result.txt
python3 PP_sim.py 1 0 > ./statistics/non_pipeline/ParallelismMapping/result.txt
python3 PP_sim.py 2 0 > ./statistics/non_pipeline/TransferMapping/result.txt

# pipeline
python3 PP_sim.py 0 1 > ./statistics/pipeline/DefaultMapping/result.txt
python3 PP_sim.py 1 1 > ./statistics/pipeline/ParallelismMapping/result.txt
python3 PP_sim.py 2 1 > ./statistics/pipeline/TransferMapping/result.txt