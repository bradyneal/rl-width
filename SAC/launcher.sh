for ((i=0;i<5;i+=1))
do 
    

    ENVIRONMENT="HalfCheetah-v1"
    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name MaxEntStateSAC --expl_type MaxEntState --use_logger --seed $i --lamda_regularizer 0.1 --lamda_ent 0.01 
    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name MaxEntStateSAC --expl_type KLD --use_logger --seed $i --lamda_regularizer 0.1 --lamda_ent 0.01 
    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name MaxEntStateSAC --expl_type StateEntropy --use_logger --seed $i --lamda_regularizer 0.1 --lamda_ent 0.01 
    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name SAC --expl_type Baseline --use_logger --seed $i --lamda_regularizer 0.1 --lamda_ent 0.01 


done

echo "ALL DONE"



for ((i=0;i<5;i+=1))
do 
    

    ENVIRONMENT="Hopper-v1"

    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name MaxEntStateSAC --expl_type MaxEntState --use_logger --seed $i --lamda_regularizer 0.1 --lamda_ent 0.01 
    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name MaxEntStateSAC --expl_type KLD --use_logger --seed $i --lamda_regularizer 0.1 --lamda_ent 0.01 
    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name MaxEntStateSAC --expl_type StateEntropy --use_logger --seed $i --lamda_regularizer 0.1 --lamda_ent 0.01 
    CUDA_VISIBLE_DEVICES=0 python main.py --env_name $ENVIRONMENT --policy_name SAC --expl_type Baseline --use_logger --seed $i --lamda_regularizer 0.1 --lamda_ent 0.01 


done

echo "ALL DONE"



