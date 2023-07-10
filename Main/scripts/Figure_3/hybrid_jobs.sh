# #!/bin/bash

# FSS Dataset

joblist=$(sq -h --format="%j")

for delta in $(seq -1.545 0.5 13.455) 
do
    # for data_epochs in 1000
    for data_epochs in 100
    do
        for seed in $(seq 111 111 111) # one seed for now
        do
            # X="d=$delta|OneD|Nh=32|Hybrid_$data_epochs|$seed"
            # sbatch -J "$X" --export="delta=$delta,data_epochs=$data_epochs,dim=OneD,nh=32,seed=$seed" submit_hybrid_training.sh

            X="d=$delta|TwoD|Nh=16|Hybrid_$data_epochs|$seed"
            sbatch -J "$X" --export="delta=$delta,data_epochs=$data_epochs,dim=TwoD,nh=16,seed=$seed" submit_hybrid_training.sh
                
        done 
        sleep 0.5s
    done
done
