# No Init Setting

#  window-close-v2
nohup   python From_scratch_LaRes_train_network.py --env-name='window-close-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max' --windows_length=500 --elite_num=3  --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/1.log 2>&1 &
nohup   python From_scratch_LaRes_train_network.py --env-name='window-close-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max' --windows_length=500 --elite_num=3  --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini --total-timesteps=1000000  --seed 2  --eval-episodes 20  --LLM_freq=200000 > ./logs/2.log 2>&1 &

# windows-open-v2
nohup   python From_scratch_LaRes_train_network.py --env-name='window-open-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max' --windows_length=200 --elite_num=3  --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/1.log 2>&1 &
nohup   python From_scratch_LaRes_train_network.py --env-name='window-open-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max' --windows_length=200 --elite_num=3  --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini --total-timesteps=1000000  --seed 2  --eval-episodes 20  --LLM_freq=200000 > ./logs/2.log 2>&1 &

#  button-press-v2 
nohup   python From_scratch_LaRes_train_network.py --env-name='button-press-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max' --windows_length=50 --elite_num=3  --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/3.log 2>&1 &
nohup   python From_scratch_LaRes_train_network.py --env-name='button-press-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max' --windows_length=50 --elite_num=3  --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini --total-timesteps=1000000  --seed 2  --eval-episodes 20  --LLM_freq=200000 > ./logs/4.log 2>&1 &

# door-close-v2 
nohup   python From_scratch_LaRes_train_network.py --env-name='door-close-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max' --windows_length=100 --elite_num=3  --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/5.log 2>&1 &
nohup   python From_scratch_LaRes_train_network.py --env-name='door-close-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max' --windows_length=100 --elite_num=3  --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini --total-timesteps=1000000  --seed 2  --eval-episodes 20  --LLM_freq=200000 > ./logs/6.log 2>&1 &

# drawer-open-v2 
nohup   python From_scratch_LaRes_train_network.py --env-name='drawer-open-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max' --windows_length=200 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/7.log 2>&1 &
nohup   python From_scratch_LaRes_train_network.py --env-name='drawer-open-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max' --windows_length=200 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini --total-timesteps=1000000  --seed 2  --eval-episodes 20  --LLM_freq=200000 > ./logs/8.log 2>&1 &





# Init setting

nohup   python LaRes_with_init.py --env-name='coffee-pull-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max'  --pop_size=5 --windows_length=20 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/4.log 2>&1 &

nohup   python LaRes_with_init.py --env-name='coffee-push-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max'  --pop_size=5 --windows_length=20 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/4.log 2>&1 &

nohup   python LaRes_with_init.py --env-name='hand-insert-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max'  --pop_size=5 --windows_length=50 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini-mini --total-timesteps=1000000  --seed 3  --eval-episodes 20  --LLM_freq=200000 > ./logs/3.log 2>&1 &

nohup   python LaRes_with_init.py --env-name='basketball-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max'  --pop_size=5 --windows_length=50 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/4.log 2>&1 &

nohup   python LaRes_with_init.py --env-name='dial-turn-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max'  --pop_size=5 --windows_length=50 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/4.log 2>&1 &

nohup   python LaRes_with_init.py --env-name='soccer-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max'  --pop_size=5 --windows_length=100 --elite_num=3  --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/1.log 2>&1 &

nohup   python LaRes_with_init.py --env-name='push-back-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max'  --pop_size=5 --windows_length=100 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/4.log 2>&1 &

nohup   python LaRes_with_init.py --env-name='pick-out-of-hole-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max'  --pop_size=5 --windows_length=200 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini --total-timesteps=1000000  --seed 3  --eval-episodes 20  --LLM_freq=200000 > ./logs/2.log 2>&1 &

nohup   python LaRes_with_init.py --env-name='hammer-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max'  --pop_size=5 --windows_length=500 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/4.log 2>&1 &

nohup   python LaRes_with_init.py --env-name='peg-unplug-side-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max'  --pop_size=5 --windows_length=500 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/4.log 2>&1 &

nohup   python LaRes_with_init.py --env-name='peg-insert-side-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max'  --pop_size=5 --windows_length=500 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/4.log 2>&1 &

nohup   python LaRes_with_init.py --env-name='button-press-topdown-v2' --buffer_transfer=0 --scale_type=2 --Inter_Loop_freq=1000000   --ucb_type='max'  --pop_size=5 --windows_length=500 --elite_num=3   --RL_ucb=1 --c=0.0 --EA_tau=0.1 -damp=1e-1  --model=gpt-4o-mini-mini --total-timesteps=1000000  --seed 1  --eval-episodes 20  --LLM_freq=200000 > ./logs/4.log 2>&1 &

