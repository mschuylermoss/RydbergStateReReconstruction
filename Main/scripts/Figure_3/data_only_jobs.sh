# #!/bin/bash

# FSS Dataset

joblist=$(sq -h --format="%j")

for delta in $(seq -1.545 0.5 13.455) 
do
    for seed in $(seq 111 111 111) # one seed for now
    do
        X="$delta|OneD|Nh=32|Data|$seed"
       	sbatch -J "$X" --export="delta=$delta,dim=OneD,nh=32,seed=$seed" submit_data_training.sh

        X="$delta|TwoD|Nh=16|Data|$seed"
       	sbatch -J "$X" --export="delta=$delta,dim=TwoD,nh=16,seed=$seed" submit_data_training.sh
            
    done 
    sleep 0.5s
done

