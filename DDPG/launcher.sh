for ((i=0;i<5;i+=1))
do 
    

    ENVIRONMENT="HalfCheetah-v1"

    CUDA_VISIBLE_DEVICES=1 python main.py --env_name $ENVIRONMENT --policy_name DDPG --expl_type Baseline --use_logger --seed $i 


done

echo "ALL DONE"



for ((i=0;i<5;i+=1))
do 

    ENVIRONMENT="Hopper-v1"

    CUDA_VISIBLE_DEVICES=1 python main.py --env_name $ENVIRONMENT --policy_name DDPG --expl_type Baseline --use_logger --seed $i 

done

echo "ALL DONE"


for ((i=0;i<5;i+=1))
do 

    ENVIRONMENT="Walker2d-v1"

    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name DDPG --expl_type Baseline --use_logger --seed $i 

done

echo "ALL DONE"






for ((i=0;i<5;i+=1))
do 
    

    ENVIRONMENT="Humanoid-v1"

    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name MaxEntStateDDPG --expl_type VSE --use_logger --seed $i 
    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name MaxEntStateDDPG --expl_type MaxEntState --use_logger --seed $i 
    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name MaxEntStateDDPG --expl_type CrossEntropy --use_logger --seed $i 
    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name DDPG --expl_type Baseline --use_logger --seed $i 

