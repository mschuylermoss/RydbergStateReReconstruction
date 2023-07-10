# #!/bin/bash

# FSS Dataset

joblist=$(sq -h --format="%j")

# for delta in $(seq -1.545 0.5 13.455) # all deltas
for delta in -1.545 4.455 4.955 13.455
do
    for seed in $(seq 200 100 200) # one seed for now
    do
        X="QMC_data1k|$delta|OneD|Nh=32|$seed"
       	sbatch -J "$X" --export="delta=$delta,dim=OneD,nh=32,seed=$seed" submit_data_training.sh

        X="QMC_data1k|$delta|TwoD|Nh=16|$seed"
       	sbatch -J "$X" --export="delta=$delta,dim=TwoD,nh=16,seed=$seed" submit_data_training.sh
            
    done 
    sleep 0.5s
done

