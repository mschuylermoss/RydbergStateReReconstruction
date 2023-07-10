# #!/bin/bash

# FSS Dataset

joblist=$(sq -h --format="%j")

# for delta in $(seq -1.545 0.5 13.455) # all deltas
for delta in -1.545 4.455 4.955 13.455
do
    
    for seed in $(seq 500 100 500) # one seed for now
    do
        X="d=$delta|OneD|Nh=32|Hybrid_1000|$seed"
        sbatch -J "$X" --export="delta=$delta,data_epochs=1000,dim=OneD,nh=32,seed=$seed,lr=1e-4" submit_hybrid_training.sh

        X="$delta|TwoD|Nh=16|Hybrid_50|$seed"
        sbatch -J "$X" --export="delta=$delta,data_epochs=100,dim=TwoD,nh=16,seed=$seed,lr=1e-3" submit_hybrid_training.sh
            
    done 
    sleep 0.5s

done

