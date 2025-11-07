import copy
import os
import torch
import wandb
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)


# If necessary, config the OPENAI_BASE_URL
# os.environ["OPENAI_BASE_URL"]=""
torch.set_num_threads(cpu_num)
import numpy as np
import random
from arguments import get_args
from sac import sac_agent
def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()
import utils
from utils import get_action_info, reward_recorder, env_wrapper, Worker,  reward_function_dict, parents_function_dict, input_dict, criteria_code_dict, task_description_dict

Ares_ROOT_DIR = os.getcwd()
import time
from openai import OpenAI

client = OpenAI(api_key="input your key first")

from datetime import datetime
from replay_buffer import replay_buffer

def _get_tensor_inputs(obs, args):
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if args.cuda else 'cpu').unsqueeze(0)
    return obs_tensor

Reward_recorder = reward_recorder(10)
Success_recorder = reward_recorder(10)


def evluate(current_index, env, Nan_Net,  actor_net, args, buffer, reward_function_pop, reward_input_name_pop, responses_list, dir_path, LLM_iter, reward_string_pop,project_name, Final_reward_scale, Final_reward_mu_elite,Final_reward_mu_new):
    ep_steps = 0
    ep_reward = 0
    done = False
    obs, _ = env.reset()
    s = time.time()

    data_list = []
    while not done:

        ep_steps += 1
        with torch.no_grad():
            obs_tensor = _get_tensor_inputs(obs, args)
            pi = actor_net(obs_tensor)
            action = get_action_info(pi, cuda=args.cuda).select_actions(reparameterize=False)
            action = action.cpu().numpy()[0]

            if np.isnan(action).any() or np.isinf(action).any():
                print(pi)
                print(action)
                print("Action contains NaN values.")
                return None, None, None, None


        # input the actions into the environment
        obs_, reward, done, info = env.step(env.action_space.high * action)
        org_info = env._env.get_dict()
        Reward_recorder.add_rewards(reward)
        ep_reward += reward
        other_rewards= []


        for index_index, rf in enumerate(reward_function_pop):
            if index_index in Nan_Net:
                other_rewards.append(0.0)
                continue
            input = []
            for name in reward_input_name_pop[index_index]:
                input.append(org_info[name])
            while True:
                try:
                   # other_rewards.append(rf(*input)[0]*Final_reward_scale[index_index])
                    other_rewards.append((rf(*input)[0] - Final_reward_mu_new[index_index]) * Final_reward_scale[index_index] + Final_reward_mu_elite)
                    break
                except Exception as e:

                    if index_index == current_index:
                        if index_index not in Nan_Net:
                            Nan_Net.append(index_index)
                        return None, None, None, None
                    else:
                        if index_index not in Nan_Net:
                            Nan_Net.append(index_index)
                    other_rewards.append(0.0)
                    with open('./logs/' + project_name + '/data.pkl', 'wb') as f:
                        pickle.dump([org_info], f)
                    stdout_str = str(e)
                    break


        other_rewards.append(reward)
        # store the samples
        buffer.add(org_info, obs, action, other_rewards, obs_, float(done))
        data_list.append(copy.deepcopy(([], obs, action, other_rewards, obs_, float(done))))
        # reassign the observations
        obs = obs_
        if done:
            Reward_recorder.start_new_episode()
            Success_recorder.add_rewards(info['success'])
            Success_recorder.start_new_episode()
    return ep_reward, ep_steps, data_list, info['success']



def get_LLM_reward_function(sample_size, model, messages, temperature):

    total_samples = 0
    total_token = 0
    total_completion_token = 0
    prompt_tokens = 0
    
    
    if "qwen" in model:
        chunk_size = 1
    else:
        chunk_size = sample_size
    responses = []
    while True:

        if total_samples >= sample_size:
            break

        for attempt in range(1000):
            try:


                response_cur = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=chunk_size
                )

                total_samples += chunk_size
                break
            except Exception as e:
                if attempt >= 10:
                    chunk_size = max(int(chunk_size / 2), 1)
                    print("Current Chunk Size", chunk_size)
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            print("Code terminated due to too many failed attempts!")
            exit()
        responses.extend(response_cur.choices)
        prompt_tokens += response_cur.usage.prompt_tokens
        total_completion_token += response_cur.usage.completion_tokens
        total_token += response_cur.usage.total_tokens

    return responses, prompt_tokens, total_completion_token, total_token
    

import logging
logging.basicConfig(filename='error.log', level=logging.ERROR)

import re
import ast
import subprocess
import pickle
import math



def get_reward_functions(dir_path, LLM_iter, args, temp_org_info, suggestions, initial_system, initial_user, task_obs_code_string_1, criteria_code_string, task_obs_code_string_2, input_dict_string, code_output_tip , RL_best=False, provided_response = None, code_feedback = None, real_num = 5):
    reward_function_pop = []
    code_string_list = []
    namespace= {}
    get_res_try_num = 0
    reward_string_pop = []
    response_list =[]


    if RL_best:
        messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user.format(
            task_obs_code_string_1=task_obs_code_string_1, criteria_code_string=criteria_code_string,
            task_obs_code_string_2=task_obs_code_string_2,
            input_dict_string=input_dict_string) + "\n" + code_feedback + "\n" + code_output_tip}]
    else:
        messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user.format(
            task_obs_code_string_1=task_obs_code_string_1, criteria_code_string=criteria_code_string,
            task_obs_code_string_2=task_obs_code_string_2,
            input_dict_string=input_dict_string) +  "\n" + code_output_tip}]
    if provided_response is not None:
        messages.extend([{"role": "system", "content": provided_response}, {"role": "user", "content": code_feedback + "\n" + code_output_tip}])
    try_num = 1
    success_num = 0
    while True:
        if success_num == real_num:
            break
        responses, prompt_tokens, total_completion_token, total_token = get_LLM_reward_function(real_num*2, args.model, messages, 1.0)

        for iiii in range(len(responses)):
            response_cur = responses[iiii].message.content
            print(f"Try generate num {try_num}", prompt_tokens, total_completion_token, total_token)
            try_num += 1
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    break  # Add this break statement

            code = "import numpy as np" + '\n' + "import reward_utils" + '\n'  + "from reward_utils import *" + '\n'+ code_string

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            temp_file_path = dir_path + "/" + timestamp + '_generated_code.py'

            with open('./test_generate_code.py', 'r') as f:
                test_code = f.read()
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                head = "import os" + "\n" +  "import sys" + "\n" + "current_dir = os.path.dirname(os.path.abspath(__file__))" + "\n" + """parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))""" + "\n" + "sys.path.insert(0, parent_dir)" + "\n"

                f.write(head)
                f.write(code)
                f.write("\n# Test code appended\n")
                f.write(test_code)

            filter_filepath =  dir_path +  f"/RF_{get_res_try_num}_response.txt"
            with open(filter_filepath, 'w') as f:
                process = subprocess.Popen(['python', '-u', temp_file_path, dir_path + "/" + "data.pkl"], stdout=f, stderr=f)
            process.communicate()

            with open(filter_filepath, 'r') as f:
                stdout_str = f.read()
            print(stdout_str)
            if "Success!" not in stdout_str:
                print("Fail ", iiii , stdout_str)
                continue
            else:
                response_txt = dir_path + "/" + "Iter_" + str(LLM_iter) + "_Response_" + str(success_num) + ".txt"
                with open(response_txt, 'w', encoding='utf-8') as f:
                    f.write(response_cur)
                pattern = r"(def\s+\w+\s*\(.*?\):[\s\S]*?)(?=\ndef\s|\Z)"
                matches = re.findall(pattern, code_string, re.DOTALL)
                response_list.append(response_cur)
                for i, match in enumerate(matches, 1):
                    if "def compute_reward" in match.strip():
                        code_1 =  match.strip()
                    if "def _gripper_caging_reward" in match.strip():
                        code_2 = match.strip()
                reward_string_pop.append([code_1, code_2])
                code_string_list.append(code)

            pattern = r"Conclusion:\s*(.*)"

            match = re.search(pattern, response_cur)
            temp_file_path = dir_path + "/" +  "Iter_"+str(LLM_iter)+"_Reward_Code_" + str(get_res_try_num) + ".py"
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            get_res_try_num += 1
            success_num += 1
            if success_num == real_num:
                break

    reward_input_name_pop = []
    for index, code_string in enumerate(code_string_list):
        namespace[str(index)] = {}
        exec(code_string, namespace[str(index)])

        tree = ast.parse(code_string)

        input_name = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                reward_args = [arg.arg for arg in node.args.args]
                if function_name == "compute_reward":
                    input_name = reward_args
        reward_input_name_pop.append(input_name)
        input = []
        for name in input_name:
            input.append(temp_org_info[5][name])
        print("Get input")
        res = namespace[str(index)]["compute_reward"](*input)
        print("Recheck ", res)
        reward_function_pop.append(namespace[str(index)]["compute_reward"])
    return reward_function_pop, reward_input_name_pop, reward_string_pop, response_list




from scipy.stats import beta

class ThompsonSampling:
    def __init__(self, n_arms, c, windows_length):
        self.n_arms = n_arms
        self.c = c
        self.successes = []
        self.failures = []
        self.windows_length = windows_length
        for i in range(n_arms):
            self.successes.append([0])
            self.failures.append([0])

    def select_arm(self, Nan_Net):
        sampled_theta = [beta.rvs(1 + np.sum(self.successes[i]), 1 + np.sum(self.failures[i])) for i in
                         range(self.n_arms)]

        for index, value in enumerate(sampled_theta):
            if index in Nan_Net:
                sampled_theta[index] = -1000000000

        return np.argmax(sampled_theta)

    def get_scores(self):
        sampled_theta = [beta.rvs(1 + np.sum(self.successes[i]), 1 + np.sum(self.failures[i])) for i in
                         range(self.n_arms)]

        return sampled_theta


    def update(self, chosen_arm, reward):
        if reward > 0.5:
            self.successes[chosen_arm].append(1)
            self.successes[chosen_arm] = self.successes[chosen_arm][-self.windows_length:]
        else:
            self.failures[chosen_arm].append(1)
            self.failures[chosen_arm] = self.failures[chosen_arm][-self.windows_length:]





import multiprocessing as mp
if __name__ == '__main__':
    args = get_args()
    # build the environment
    parents_code = parents_function_dict[args.env_name]
    name = "Pop_size_"+str(args.pop_size)+"_elite_num_"+str(args.elite_num)+"_Total_Right_"+ str(args.model)+"_TS_"+str(args.windows_length) +"_win_rate_Q_Pi_Net_mean_std_scale_distance_1.0_fixed_elite_num_"+str(args.elite_num)+"_Inner_Loop_"+str(args.Inter_Loop_freq)+"_Buffer_transfer_"+str(args.buffer_transfer)+"_No_sugg_"+ str(args.model)+"_Right_"+str(args.Type)+"_ARes_"+str(args.LLM_freq)+"_SAC_Env_"+ str(args.pop_size) + "_" + str(args.batch_size) + "_" + str(args.env_name) + "_steps_" + str(args.total_timesteps)
    our_wandb = wandb.init(project="Ares-MetaWorld-v2", name=name)

    name  = str(args.seed) + "_" + name
    if not os.path.exists("./logs/" + name):
        os.makedirs("./logs/" + name)
    env = utils.make_metaworld_env(args, args.seed)
    env = env_wrapper(env, args)
    # create the eval env
    eval_env = utils.make_metaworld_env(args, args.seed + 100)
    eval_env = env_wrapper(eval_env, args)
    # set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    # set the seed of torch
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    prompt_dir = f'{Ares_ROOT_DIR}/utils/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/new_code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    human_designed_code_feedback = file_to_string(f'{prompt_dir}/human_designed_code_feedback.txt')

    initial_user = file_to_string(f'{prompt_dir}/new_initial_user.txt')

    task_description = task_description_dict[args.env_name] # "picking up a stick and inserting it into a square hole in the wall"

    task_obs_code_string_1 = reward_function_dict[args.env_name]

    criteria_code_string =  criteria_code_dict[args.env_name]

    task_obs_code_string_2 = parents_code

    input_dict_string = input_dict[args.env_name]

    initial_system = initial_system.format(task_target=task_description)
    # create the agent
    sac_trainer = sac_agent(args.model, args.env_name, env, eval_env, args, our_wandb)

    global_timesteps = 0

    temp_org_info = sac_trainer._initial_exploration(exploration_policy=args.init_exploration_policy)
    print(temp_org_info[0])

    with open('./logs/'+name+'/data.pkl', 'wb') as f:
        pickle.dump(temp_org_info, f)


    use_LLM_for_Reward_search_freq = args.LLM_freq

    LLM_generate_num = 0

    memory_path = "./logs/"+ name + "/memory.pkl"
    model_path = "./logs/" + name + "/"

    manager = mp.Manager()
    queue = manager.Queue(args.pop_size + 1)
    parameters = manager.dict()

    pop = []
    for _ in range(args.pop_size):
        pop.append(sac_agent(args.model, args.env_name, env, eval_env, args, our_wandb))

    all_individual =  pop +  [sac_trainer]


    reward_function_pop, reward_input_name_pop, reward_string_pop, responses_list = get_reward_functions("./logs/" + name , LLM_generate_num, args, temp_org_info, None, initial_system, initial_user, task_obs_code_string_1,criteria_code_string, task_obs_code_string_2, input_dict_string, code_output_tip,real_num=args.pop_size)




    # before the official training, do the initial exploration to add episodes into the replay buffer
    # reset the environment
    All_buffer = replay_buffer(args.buffer_size)
    previous_print = 0
    previous_LLM = 0
    sac_trainer._initial_exploration_and_store(All_buffer, reward_function_pop, reward_input_name_pop, exploration_policy=args.init_exploration_policy)

    worker_pop = []
    for p_id in range(len(all_individual)):

        wok = Worker(All_buffer, p_id, args.buffer_size, memory_path, model_path, args.target_update_interval,  args.seed, parameters, queue, args.model, args.env_name, env, eval_env, args)
        worker_pop.append(wok)
        wok.start()

    update_model_flag = 0
    best_log_alpha = None
    global_best_index_num = -1
    reward_refine = []
    Nan_Net = []
    global_best_one = -1


    times_recorder = {}
    for index in range(args.pop_size+1):
        times_recorder[index] = 0
    total_inter_num= 0

    TS = ThompsonSampling(len(pop), args.c, args.windows_length)

    reset_ucb_time_steps = 0
    Final_reward_scale = []
    Final_reward_mu_new = []
    for _ in range(args.pop_size+1):
        Final_reward_scale.append(1.0)
        Final_reward_mu_new.append(0.0)

    Final_reward_mu_elite = 0.0

    RL_inter_num = 0
    EA_inter_num = 0
    while global_timesteps < args.total_timesteps:

        if global_timesteps - previous_LLM >= use_LLM_for_Reward_search_freq:
            # start to use LLM to search for the reward function
            times_recorder = {}
            for index in range(args.pop_size + 1):
                times_recorder[index] = 0
            TS = ThompsonSampling(len(pop), args.c, args.windows_length)
            reset_ucb_time_steps = global_timesteps
            total_inter_num = 0
            t1 = time.time()
            all_data_temp = []
            for d  in All_buffer.storge:
                all_data_temp.append(d[0])

            with open('./logs/' + name + '/data.pkl', 'wb') as f:
                pickle.dump(all_data_temp, f)
            print("Save all data", time.time()-t1)
            all_individual_success_rates = []
            all_individual_rewards = []
            all_individual_infos = []
            all_individual_own_rewards = []
            ea_num = len(pop)
            for index, ea_agent in enumerate(pop+ [sac_trainer]):

                if index in Nan_Net:
                    all_individual_own_rewards.append(-1000000)
                    mean_rewards, mean_success, all_infos = -1000000, 0.0, ["Nan or Inf"]
                else:
                    if index == ea_num:
                        mean_rewards, mean_success, all_infos, _ = sac_trainer._evaluate_agent(ea_agent.actor_net)
                        all_individual_own_rewards.append(mean_rewards)
                    else:
                        mean_rewards, mean_success, all_infos, mean_own_reward = sac_trainer._evaluate_agent(ea_agent.actor_net,  reward_function_pop[index], reward_input_name_pop[index])
                        all_individual_own_rewards.append(mean_own_reward)
                all_individual_success_rates.append(mean_success)
                all_individual_rewards.append(mean_rewards)
                all_individual_infos.append(all_infos)


            print("all_individual_success_rates", all_individual_success_rates)
            print("all_individual_rewards", all_individual_rewards)
            print("all_individual_own_rewards", all_individual_own_rewards)
            if np.mean(all_individual_success_rates) == 0:
                best_code_index = np.argmax(all_individual_rewards)
                elite_index_list = np.argsort(all_individual_rewards)[-args.elite_num:][::-1]

                rrrrank = np.argsort(all_individual_rewards)[::-1]
            else:
                best_code_index = np.argmax(all_individual_success_rates)
                elite_index_list = np.argsort(all_individual_success_rates)[-args.elite_num:][::-1]
                rrrrank = np.argsort(all_individual_success_rates)[::-1]
            global_best_one = best_code_index
            our_wandb.log(
                {'RL_inter_num': RL_inter_num, 'EA_inter_num': EA_inter_num, 'Best_code_index':best_code_index,'Best_policy_win_rate': np.max(all_individual_success_rates), 'Best_policy_reward': np.max(all_individual_rewards),
                 'Best_EA_policy_win_rate': np.max(all_individual_success_rates[:ea_num]),
                 'Best_EA_policy_reward': np.max(all_individual_rewards[:ea_num]), 'LLM_generate_num': LLM_generate_num, 'time_steps': global_timesteps})
            LLM_generate_num +=1
            best_info = all_individual_infos[best_code_index]
            previous_LLM = global_timesteps

            file_content = "Current time steps " + str(global_timesteps) + "\n"
            file_content += "Best index " + str(best_code_index) + "  \n"
            file_content += "Rank order" + str(rrrrank) + "\n"

            for iiii, string_r in enumerate(reward_string_pop):
                file_content += "============================"
                file_content += str(iiii)+"_" + str(string_r) + "\n"
                file_content += str(responses_list[iiii]) + "\n"

            with open("./logs/"+name + "_" + str(global_timesteps) +".txt", "w", encoding="utf-8") as file:
                file.write(file_content)

            our_wandb.save("./logs/"+name + "_" + str(global_timesteps) +".txt") 

            if best_code_index == len(pop):
                # get new reward function
                current_code_feedback = human_designed_code_feedback.format(win_rate=str(all_individual_success_rates[best_code_index]),  current_our_score=str(all_individual_rewards[-1]),  current_output= str(best_info))
                temp_reward_function_pop, temp_reward_input_name_pop, temp_reward_string_pop, temp_responses_list = get_reward_functions(
                    "./logs/" + name, LLM_generate_num, args, temp_org_info, None, initial_system,
                    initial_user, task_obs_code_string_1, criteria_code_string, task_obs_code_string_2,input_dict_string, code_output_tip, True, None, "We train the RL policy for " + str(args.LLM_freq) + " environment steps. \n"+current_code_feedback, real_num= len(pop)+1 - args.elite_num)
            else:
                current_code_feedback = code_feedback.format(
                    win_rate=str(all_individual_success_rates[best_code_index]),
                    current_score=str(all_individual_rewards[-1]),
                    current_our_score=str(all_individual_own_rewards[best_code_index]), current_output=str(best_info))

                # get new reward function
                temp_reward_function_pop, temp_reward_input_name_pop, temp_reward_string_pop, temp_responses_list = get_reward_functions("./logs/" + name , LLM_generate_num, args, temp_org_info,None,initial_system,
                                                                                                     initial_user,task_obs_code_string_1,criteria_code_string, task_obs_code_string_2,
                                                                                                     input_dict_string,code_output_tip,  False, responses_list[best_code_index] , "We train the RL policy for " + str(args.LLM_freq) + " environment steps. \n"+current_code_feedback, real_num= len(pop)+1 - args.elite_num)


            agent_index_list = list(range(len(pop)+1))
            no_elite_list = list(set(agent_index_list) - set(elite_index_list))

            if len(pop) in no_elite_list:
                no_elite_list.remove(len(pop))

            for iii, index in enumerate(no_elite_list):
                reward_function_pop[index] = temp_reward_function_pop[iii]
                reward_input_name_pop[index] = temp_reward_input_name_pop[iii]
                reward_string_pop[index] = temp_reward_string_pop[iii]
                responses_list[index] = temp_responses_list[iii]

            Nan_Net = []

            reward_list_dict={}
            new_reward_list_dict = {}
            for _ in range(args.pop_size + 1):
                reward_list_dict[_] = []
                new_reward_list_dict[_] = []


            for data_index, single_data in enumerate(All_buffer.storge):
                org_info, obs, action, reward_list, obs_, done = single_data
                for _, rrr in enumerate(reward_list):
                    if math.isnan(rrr) or math.isinf(rrr):
                        continue
                    reward_list_dict[_].append(rrr)

                for index_index, rf in enumerate(reward_function_pop):
                    if index_index in elite_index_list:
                        new_reward_list_dict[index_index].append(reward_list[index_index])
                    else:
                        input = []
                        for input_name in reward_input_name_pop[index_index]:
                            input.append(org_info[input_name])
                        new_reward_list_dict[index_index].append(rf(*input)[0])

                new_reward_list_dict[args.pop_size].append(reward_list[-1])

            Elite_reward_scale = []
            expert_reward_std = np.std(reward_list_dict[args.pop_size])
            best_reward_std =  np.std(reward_list_dict[int(best_code_index)])
            Final_reward_mu_elite = np.mean(reward_list_dict[int(best_code_index)])
            our_wandb.log({'Scale/Individual_elite_std': best_reward_std,
                            'Scale/Individual_elite_mean': Final_reward_mu_elite, 'time_steps': global_timesteps})
            for ___ in range(len(pop)+1):
                if len(reward_list_dict[___]) == 0:
                    continue
                std_dev = np.std(reward_list_dict[___])
                curr_max =  np.max(reward_list_dict[___])
                curr_min = np.min(reward_list_dict[___])
                curr_mean = np.mean(reward_list_dict[___])

                new_std_dev = np.std(new_reward_list_dict[___])
                new_curr_max =  np.max(new_reward_list_dict[___])
                new_curr_min = np.min(new_reward_list_dict[___])
                new_curr_mean = np.mean(new_reward_list_dict[___])


                Final_reward_mu_new[___] = new_curr_mean

                Elite_reward_scale.append(best_reward_std/new_std_dev)

                our_wandb.log({'Scale/Individual_'+ str(___) + "_Reward_std": std_dev, 'Scale/Individual_'+ str(___) + "_Reward_mean": curr_mean,'Scale/Individual_'+ str(___) + "_Reward_max": curr_max, 'Scale/Individual_'+ str(___) + "_Reward_min": curr_min,'time_steps': global_timesteps})
                our_wandb.log({'Scale/scale_elite_'+ str(___) + "_std": best_reward_std/new_std_dev, 'Scale/scale_expert'+ str(___) + "_std": expert_reward_std/new_std_dev, 'Scale/scale_pre_and_new' + str(___) + "_std": std_dev/new_std_dev,'Scale/Individual_' + str(___) + "_New_Reward_std": new_std_dev,
                               'Scale/Individual_' + str(___) + "_New_Reward_mean": new_curr_mean,
                               'Scale/Individual_' + str(___) + "_New_Reward_max": new_curr_max,
                               'Scale/Individual_' + str(___) + "_New_Reward_min": new_curr_min, 'time_steps': global_timesteps})

            # data refresh
            print("Refresh all data start.....")
            Final_reward_scale = Elite_reward_scale

            # for all elites，do not scale
            for iiii in elite_index_list:
                Final_reward_mu_new[iiii] = Final_reward_mu_elite
                Final_reward_scale[iiii] =  1.0

            for data_index, single_data in enumerate(All_buffer.storge):
                org_info, obs, action, reward_list, obs_, done = single_data
                other_rewards = []

                if args.buffer_transfer == 1:
                    for index_index, rf in enumerate(reward_function_pop):
                        other_rewards.append(reward_list[best_code_index])
                    other_rewards.append(reward_list[best_code_index])
                elif args.buffer_transfer == 0:
                    for index_index, rf in enumerate(reward_function_pop):
                        if index_index in Nan_Net:
                            continue
                        if index_index in elite_index_list:
                            other_rewards.append(reward_list[index_index])
                        else:
                            input = []
                            for input_name in reward_input_name_pop[index_index]:
                                input.append(org_info[input_name])
                            try:
                                other_rewards.append((rf(*input)[0] - Final_reward_mu_new[index_index])* Final_reward_scale[index_index] + Final_reward_mu_elite)
                            except Exception as e:
                                print("!!!! Big error !!! Data refresh error", e)
                                print(index_index, Final_reward_scale)
                                print(Nan_Net)
                                exit()
                                Nan_Net.append(index_index)
                                other_rewards.append(0.0)
                    other_rewards.append(reward_list[-1])

                reward_refine.append(other_rewards)
                All_buffer.storge[data_index] = copy.deepcopy((org_info, obs, action, other_rewards, obs_, done))
            print("Data Refresh done .....")

            print("Refresh all agent start.....")

            torch.save(all_individual[best_code_index].actor_net.state_dict(), model_path  + "/best_actor_net.pth")
            torch.save(all_individual[best_code_index].qf1.state_dict(), model_path + "/best_qf1.pth")
            torch.save(all_individual[best_code_index].qf2.state_dict(), model_path + "/best_qf2.pth")
            torch.save(all_individual[best_code_index].target_qf1.state_dict(), model_path + "/best_target_qf1.pth")
            torch.save(all_individual[best_code_index].target_qf2.state_dict(), model_path + "/best_target_qf2.pth")
            torch.save(all_individual[best_code_index].actor_optim.state_dict(), model_path + "/best_actor_optim.pth")
            torch.save(all_individual[best_code_index].qf1_optim.state_dict(), model_path + "/best_qf1_optim.pth")
            torch.save(all_individual[best_code_index].qf2_optim.state_dict(), model_path + "/best_qf2_optim.pth")
            best_log_alpha = copy.deepcopy(all_individual[best_code_index].log_alpha.detach().clone().data)

            print("get current best_log_alpha", best_log_alpha)
            global_best_index_num = elite_index_list
            print("Refresh all agent done.....")
            update_model_flag = 1
            RL_inter_num = 0
            EA_inter_num = 0


        if global_timesteps - reset_ucb_time_steps > args.Inter_Loop_freq:
            reset_ucb_time_steps = global_timesteps
            
            times_recorder = {}
            for index in range(args.pop_size + 1):
                times_recorder[index] = 0
            total_inter_num = 0

        total_ep_reward = 0
        all_data_list = []
        fitness = {}

        inter_list = []
        temp_best = []
        for indexaaa in range(len(pop)):
            worker_idx = TS.select_arm(Nan_Net)

            if worker_idx in Nan_Net:
                print("Another big error?????", worker_idx, Nan_Net)
                exit()
            if worker_idx == len(pop):
                inter_list.append(len(pop))
                continue
            if worker_idx in inter_list:
                continue
            inter_list.append(worker_idx)
            ea_agent = pop[worker_idx]
            EA_inter_num +=1
            times_recorder[worker_idx] = times_recorder[worker_idx]  + 1
            total_inter_num +=1

            ep_reward, ep_steps, data_list, success = evluate(worker_idx, env, Nan_Net, ea_agent.actor_net, args, All_buffer, reward_function_pop, reward_input_name_pop, responses_list, "./logs/" + name , LLM_generate_num, reward_string_pop, name, Final_reward_scale, Final_reward_mu_elite,Final_reward_mu_new)
            if ep_reward is None:
                Nan_Net.append(worker_idx)
                continue
            all_data_list +=data_list
            total_ep_reward += ep_steps
            fitness[worker_idx] = ep_reward

            TS.update(worker_idx, float(success))
            temp_best.append(ep_reward)
            print(worker_idx, " worker ",ep_reward)

        rl_ep_reward, ep_steps, data_list, success = evluate(len(pop), env,Nan_Net, sac_trainer.actor_net, args, All_buffer, reward_function_pop, reward_input_name_pop, responses_list, "./logs/" + name , LLM_generate_num, reward_string_pop, name, Final_reward_scale, Final_reward_mu_elite,Final_reward_mu_new)
        RL_inter_num +=1
        all_data_list += data_list
        total_ep_reward += ep_steps
        fitness[len(pop)] = rl_ep_reward
        print(global_timesteps, " RL reward", rl_ep_reward)

        if len(temp_best) > 0:
            print(global_timesteps, " EA best ", np.max(temp_best))

        all_scores = TS.get_scores()
        print("fitness", fitness)
        for _, score in enumerate(all_scores):
            print(_, " index ", score)

        global_timesteps += total_ep_reward

        t1 = time.time()# save data

        for agent_index,  _ in enumerate(worker_pop):
            if agent_index not in Nan_Net:
                parameters[agent_index] = (len(All_buffer.storge), total_ep_reward,all_data_list, All_buffer.next_idx, update_model_flag, best_log_alpha, global_best_index_num, reward_refine, global_best_one)
            else:
                parameters[agent_index] = (len(All_buffer.storge), 0, all_data_list, All_buffer.next_idx, update_model_flag,best_log_alpha, global_best_index_num, reward_refine, global_best_one)

        best_log_alpha = None
        reward_refine = []

        if global_timesteps - previous_print > args.display_interval:
            # start to do the evaluation

            best_EA_rewards = []
            best_EA_win_rates = []
            for index, ea_agent in enumerate(pop):

                mean_rewards, mean_success, _ , _= sac_trainer._evaluate_agent(ea_agent.actor_net)
                if mean_rewards is None:
                    if index not in Nan_Net:
                        Nan_Net.append(index)
                    mean_rewards = -100000
                    mean_success = 0.0
                best_EA_rewards.append(mean_rewards)
                best_EA_win_rates.append(mean_success)


                agent_string = "EA_" + str(index) +"_"
                our_wandb.log({ agent_string+'Rewards': mean_rewards, agent_string+'Success': mean_success,'time_steps': global_timesteps})

            EA_best_rewards = np.max(best_EA_rewards)
            EA_best_success = np.max(best_EA_win_rates)
            our_wandb.log({'Nan_num':len(Nan_Net), 'EA_Rewards': EA_best_rewards, 'EA_Success': EA_best_success, 'time_steps': global_timesteps})
            print('[{}] Frames: {}, EA Rewards: {:.3f}, Success: {:.3f}'.format(
                datetime.now(), global_timesteps, EA_best_rewards, EA_best_success))

        # after collect the samples, start to update the network
        t1 = time.time()

        for _ in range(len(all_individual)- len(Nan_Net)):
            tt = time.time()
            agent_index, actor_state_dict, qf1_state_dict, qf2_state_dict, q1_target_state_dict, q2_target_state_dict, actor_opt_state_dict, q1_opt_state_dict, q2_opt_state_dict, qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss , current_alpha, actor_fau, Q1_fau, Q2_fau , KL_loss, kl_q1_loss, kl_q2_loss, time_cost_list, total_train_time, train_sub_space_time = queue.get()

            t2 = time.time()

            all_individual[agent_index].actor_net.load_state_dict(actor_state_dict)
            all_individual[agent_index].qf1.load_state_dict(qf1_state_dict)
            all_individual[agent_index].qf2.load_state_dict(qf2_state_dict)
            all_individual[agent_index].target_qf1.load_state_dict(q1_target_state_dict)
            all_individual[agent_index].target_qf2.load_state_dict(q2_target_state_dict)
            all_individual[agent_index].actor_optim.load_state_dict(actor_opt_state_dict)
            all_individual[agent_index].qf1_optim.load_state_dict(q1_opt_state_dict)
            all_individual[agent_index].qf2_optim.load_state_dict(q2_opt_state_dict)
            all_individual[agent_index].log_alpha.data = current_alpha

            if agent_index == len(pop):
                agent_string = "RL"
            else:
                agent_string = "EA_" + str(agent_index)

            if agent_index in fitness.keys():
                our_wandb.log(
                    {agent_string + '/Learning_Reward': fitness[agent_index],
                     'time_steps': global_timesteps})

            our_wandb.log({agent_string + "_q1_KL": kl_q1_loss,  agent_string + "_q2_KL": kl_q2_loss, agent_string + "_KL": KL_loss, agent_string +'_selected_ratio': times_recorder[agent_index]/total_inter_num, 'total_inter_num':total_inter_num,agent_string + '/actor_fau': actor_fau, agent_string + '/Q1_fau': Q1_fau, agent_string + '/Q2_fau': Q2_fau, agent_string +'/Q_loss': qf1_loss, agent_string+'/Actor_loss': actor_loss, agent_string+'/Alpha_loss': alpha_loss, agent_string+'/Alpha': alpha,'time_steps': global_timesteps})
        print("Avg train time ", (time.time()-t1)/total_ep_reward)
        update_model_flag = 0

        if global_timesteps - previous_print > args.display_interval:
            previous_print = global_timesteps
            # start to do the evaluation
            mean_rewards, mean_success, _, _ = sac_trainer._evaluate_agent(sac_trainer.actor_net)


            if mean_success > EA_best_success:
                ea_better = 0.0
            else:
                ea_better = 1.0

            our_wandb.log(
                {'EA_better_ratio': ea_better,
                 'Rewards': np.max([mean_rewards, EA_best_rewards]), 'Success': np.max([EA_best_success, mean_success]),
                 'RL_Rewards': mean_rewards, 'RL_Success': mean_success, 'T_Reward': Reward_recorder.mean,
                 'Q_loss': qf1_loss, 'Actor_loss': actor_loss, 'Alpha_loss': alpha_loss, 'Alpha': alpha,
                 'time_steps': global_timesteps})
            print(
                '[{}] Frames: {}, RL ewards: {:.3f}, Success: {:.3f}, T_Reward: {:.3f}, QF1: {:.3f}, QF2: {:.3f}, AL: {:.3f}, Alpha: {:.3f}, AlphaL: {:.3f}'.format(
                    datetime.now(), \
                    global_timesteps, mean_rewards, mean_success, Reward_recorder.mean, qf1_loss, qf2_loss,
                    actor_loss, alpha, alpha_loss))

            torch.save(sac_trainer.actor_net.state_dict(),
                       sac_trainer.model_path + '/model.pt' if args.random_init else sac_trainer.model_path + '/fixed_model.pt')
            if mean_success == 1:
                torch.save(sac_trainer.actor_net.state_dict(),
                           sac_trainer.model_path + '/best_model.pt' if args.random_init else sac_trainer.model_path + '/fixed_best_model.pt')

    for _, worker in enumerate(worker_pop):
        worker.terminate()
        worker.join()
        del worker