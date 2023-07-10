# #!/bin/bash

# FSS Dataset

joblist=$(sq -h --format="%j")

# for delta in $(seq -1.545 0.5 13.455) # all deltas
for delta in -1.545 4.455 4.955 13.455
do
    for data_epochs in $(seq 50 50 100)
    # for data_epochs in 1000 5000
    do
        for seed in $(seq 200 100 200) # one seed for now
        do
            X="QMC1k_H|$delta|OneD|Nh=32|$seed"
            sbatch -J "$X" --export="delta=$delta,data_epochs=$data_epochs,dim=OneD,nh=32,seed=$seed" submit_hybrid_training.sh

            X="QMC1k_H|$delta|TwoD|Nh=16|$seed"
            sbatch -J "$X" --export="delta=$delta,data_epochs=$data_epochs,dim=TwoD,nh=16,seed=$seed" submit_hybrid_training.sh
                
        done 
        sleep 0.5s
    done
done

