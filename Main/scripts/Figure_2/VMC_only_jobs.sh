# #!/bin/bash

# FSS Dataset

joblist=$(sq -h --format="%j")

# for delta in $(seq -1.545 0.5 13.455) # all deltas
for delta in -1.545 4.455 4.955 13.455
do
    for seed in $(seq 500 100 500) # one seed for now
    do
        X="$delta|OneD|Nh=32|VMC|$seed"
       	sbatch -J "$X" --export="delta=$delta,dim=OneD,nh=32,seed=$seed,lr=1e-3" submit_VMC_training.sh

        X="$delta|TwoD|Nh=16|VMC|$seed"
       	sbatch -J "$X" --export="delta=$delta,dim=TwoD,nh=16,seed=$seed,lr=1e-3" submit_VMC_training.sh
            
    done 
    sleep 0.5s
done

