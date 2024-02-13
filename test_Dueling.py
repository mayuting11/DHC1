'''create test set and test model'''
import random
import pickle
import multiprocessing as mp
from typing import Union
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment_D3QTP import Environment
from model_Dueling import Network
import configs_D3QTP
from tqdm import tqdm
import numpy as np
import neptune

torch.manual_seed(configs_D3QTP.test_seed)
np.random.seed(configs_D3QTP.test_seed)
random.seed(configs_D3QTP.test_seed)
test_num = 200
device = torch.device('cpu')
torch.set_num_threads(1)


def create_test(test_env_settings, num_test_cases):
    for map_length, map_width, num_agents in test_env_settings:

        name = './test_set/{}length_{}width_{}agents.pth'.format(map_length, map_width, num_agents)
        print('-----{}length {}agents {}density-----'.format(map_length, map_width, num_agents))
        tests = []
        env = Environment(num_agents=num_agents, map_length=map_length, map_width=map_width)

        # set progress bar
        for _ in tqdm(range(num_test_cases)):
            tests.append((np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos)))
            env.reset(num_agents=num_agents, map_length=map_length, map_width=map_width,
                      num_goals=configs_D3QTP.goal_num, obstacle_list=configs_D3QTP.obstacle_list,
                      goals_list=configs_D3QTP.goal_list)
        print()

        # write tests into f
        with open(name, 'wb') as f:
            pickle.dump(tests, f)


def test_model(model_range: Union[int, tuple]):
    run = neptune.init_run(
        project="yuting/DHC",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmOWRlMmNmNy00ZjI1LTRmN2ItYWUyNi0yMzEwNzYxZThiMDQifQ==",
    )
    params = {
        "lr": 1e-4,
        "bs": 192,
    }
    run["parameters"] = params

    '''
    test model in 'models' file with model number
    '''
    network = Network()
    network.eval()
    network.to(device)
    test_set = configs_D3QTP.test_env_settings
    pool = mp.Pool(mp.cpu_count())

    if isinstance(model_range, int):
        state_dict = torch.load('./models/{}.pth'.format(model_range), map_location=device)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()
        print('----------test model {}----------'.format(model_range))

        for case in test_set:
            print("test set: {} length {} width {} agents".format(case[0], case[1], case[2]))
            with open('./test_set/{}length_{}width_{}agents.pth'.format(case[0], case[1], case[2]), 'rb') as f:
                tests = pickle.load(f)

            tests = [(test, network) for test in tests]
            ret = pool.map(test_one_case, tests)

            success = 0
            avg_step = 0
            avg_reward = 0
            for i, j, k in ret:
                success += i
                avg_step += j
                avg_reward += k

            print("success rate: {:.2f}%".format(success / len(ret) * 100))
            run["train/batch/success"].append(success / len(ret) * 100)
            print("average step: {}".format(avg_step / len(ret)))
            run["train/batch/step"].append(avg_step / len(ret))
            print("average reward: {}".format(avg_reward / len(ret)))
            run["train/batch/reward"].append(avg_reward / len(ret))
            print()

    elif isinstance(model_range, tuple):
        for model_name in range(model_range[0], model_range[1] + 1, configs_D3QTP.save_interval):
            state_dict = torch.load('./models/{}.pth'.format(model_name), map_location=device)
            network.load_state_dict(state_dict)
            network.eval()
            network.share_memory()
            print('----------test model {}----------'.format(model_name))

            for case in test_set:
                print("test set: {} length {} width {} agents".format(case[0], case[1], case[2]))
                with open('./test_set/{}length_{}width_{}agents.pth'.format(case[0], case[1], case[2]), 'rb') as f:
                    tests = pickle.load(f)

                tests = [(test, network) for test in tests]
                ret = pool.map(test_one_case, tests)

                success = 0
                avg_step = 0
                avg_reward = 0
                for i, j, k in ret:
                    success += i
                    avg_step += j
                    avg_reward += k

                print("success rate: {:.2f}%".format(success / len(ret) * 100))
                run["train/batch/success"].append(success / len(ret) * 100)
                print("average step: {}".format(avg_step / len(ret)))
                run["train/batch/step"].append(avg_step / len(ret))
                print("average reward: {}".format(avg_reward / len(ret)))
                run["train/batch/reward"].append(avg_reward / len(ret))
                print()
            print('\n')


def test_one_case(args):
    run = neptune.init_run(
        project="yuting/DHC",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmOWRlMmNmNy00ZjI1LTRmN2ItYWUyNi0yMzEwNzYxZThiMDQifQ==",
    )
    params = {
        "lr": 1e-4,
        "bs": 192,
    }
    run["parameters"] = params

    env_set, network = args
    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    obs, pos, obs_sec = env.observe()
    done = False
    network.reset()
    step = 0
    reward = 0
    while not done and env.steps < configs_D3QTP.max_episode_length:
        actions, _, _ = network.step(torch.as_tensor(obs.astype(np.float32)),
                                     torch.as_tensor(obs_sec.astype(np.float32))
                                     )
        (obs, pos, obs_sec), rewards, done, _ = env.step(actions)
        step += 1
        reward += np.mean(rewards)
        run["train/batch/reward"].append(reward)
        run["train/batch/step"].append(step)
    success = 0
    for i in range(env_set[2]):
        if env.agents_pos[i] in env.goals_pos:
            success += 1
    run["train/batch/success"].append(success/env_set[2])
    return success/env_set[2], step, reward


def make_animation(model_name: int, test_set_name: tuple, test_case_idx: int, steps: int = 25):
    '''
    visualize running results
    model_name: model number in 'models' file
    test_set_name: (length, width, num_agents)
    test_case_idx: int, the test case index in test set
    steps: how many steps to visualize in test case
    '''
    color_map = np.array([[255, 255, 255],  # white: available space
                          [190, 190, 190],  # gray: obstacle space
                          [0, 191, 255],  # blue: agent starting positions
                          [255, 165, 0],  # orange: goal waypoints
                          [0, 250, 154],  # green: agents after arriving at goal waypoints
                          [255, 48, 48],  # red: hazard waypoints
                          [255, 215, 0]])  # supply waypoints

    network = Network()
    network.eval()
    network.to(device)
    state_dict = torch.load('models/{}.pth'.format(model_name), map_location=device)
    network.load_state_dict(state_dict)

    test_name = 'test_set/135_length_19_width_38_agents.pkl'
    with open(test_name, 'rb') as f:
        tests = pickle.load(f)

    env = Environment()
    env.load(tests[test_case_idx][0], tests[test_case_idx][1], tests[test_case_idx][2])
    fig = plt.figure()
    done = False
    obs, pos, obs_sec = env.observe()

    imgs = []
    while not done and env.steps < steps:
        imgs.append([])
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if env.agents_pos[agent_id] in env.goals_pos:
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
        for goal_id in range(configs_D3QTP.goal_num):
            map[tuple(env.goals_pos[goal_id])] = 3
        for supply_id in range(configs_D3QTP.supply_num):
            map[tuple(env.supply_pos[supply_id])] = 6
        for i in env.hazard_dic:
            if env.hazard_dic[i] == 2.5:
                map[i] = 5
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)

        imgs[-1].append(img)

        for i, (agent_x, agent_y) in enumerate(env.agents_pos):
            text = plt.text(agent_x, agent_y, i, color='black', ha='center', va='center')
            imgs[-1].append(text)
        for i, (goal_x, goal_y) in enumerate(env.goals_pos):
            text = plt.text(goal_x, goal_y, i, color='black', ha='center', va='center')
            imgs[-1].append(text)

        actions, _, _, = network.step(torch.from_numpy(obs.astype(np.float32)).to(device),
                                      torch.from_numpy(pos.astype(np.float32)).to(device),
                                      torch.as_tensor(obs_sec.astype(np.float32))
                                      )
        (obs, pos, obs_sec), rewards, done, _ = env.step(actions)
        # print(done)

    if done and env.steps < steps:
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if env.agents_pos[agent_id] in env.goals_pos:
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
        for goal_id in range(configs_D3QTP.goal_num):
            map[tuple(env.goals_pos[goal_id])] = 3
        for supply_id in range(configs_D3QTP.supply_num):
            map[tuple(env.supply_pos[supply_id])] = 6
        for i in env.hazard_dic:
            if env.hazard_dic[i] == 2.5:
                map[i] = 5

        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)
        for _ in range(steps - env.steps):
            imgs.append([])
            imgs[-1].append(img)
            for i, (agent_x, agent_y) in enumerate(env.agents_pos):
                text = plt.text(agent_x, agent_y, i, color='black', ha='center', va='center')
                imgs[-1].append(text)
            for i, (goal_x, goal_y) in enumerate(env.goals_pos):
                text = plt.text(goal_x, goal_y, i, color='black', ha='center', va='center')
                imgs[-1].append(text)

    ani = animation.ArtistAnimation(fig, imgs, interval=600, blit=True, repeat_delay=1000)

    ani.save('videos/{}_{}_{}_{}.mp4'.format(model_name, *test_set_name))


if __name__ == '__main__':

    create_test(test_env_settings=configs_D3QTP.test_env_settings, num_test_cases=configs_D3QTP.num_test_cases)
    test_model((2000, 6000))
    make_animation(2000, (135, 19, 38), 0)
