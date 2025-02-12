sleep 3h;
export CUDA_VISIBLE_DEVICES=0
# get running time for AntBulletEnv-v0
python main.py --seed 1 --env-name AntBulletEnv-v0 --num-env-steps 1000000 \
    --log-dir logs/time/ant/ --save-dir logs/time/ant/ \
    --device cuda:0 > logs/bs_time.log 2>&1 &
wait

python main_rr.py --seed 1 --env-name AntBulletEnv-v0 --num-env-steps 1000000 \
    --log-dir logs/time/ant/ --save-dir logs/time/ant/ \
    --device cuda:0 > logs/rr_time.log 2>&1 &