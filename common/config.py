import os
import torch
import datetime


class Config:
    # common config
    random_seed = 27  # set random seed if required (0 = no random seed)
    train_time = str(datetime.datetime.now().replace(microsecond=0).strftime("%Y-%m-%d-%H-%M-%S"))

    # cuda config
    device = torch.device('cpu')
    device_name = "cpu"
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        device_name = torch.cuda.get_device_name(device)

    # env config
    env_name = "CartPole-v1"

    # train dqn config
    memory_size = 10000
    gamma = 0.99
    max_step = 10000000
    lr = 2e-4
    batch_size = 32

    update_target_interval = 100
    print_interval = update_target_interval
    log_interval = update_target_interval

    epsilon_max = 1
    epsilon_min = 0.05
    eps_decay = 30000

    # save config
    if not os.path.exists("runs"):
        os.mkdir("runs")
    if not os.path.exists("runs/" + env_name + "_DQN"):
        os.mkdir("runs/" + env_name + "_DQN")
    reward_writer_path = "runs/" + env_name + "_DQN/" + train_time + "_train-reward"
    loss_writer_path = "runs/" + env_name + "_DQN/" + train_time + "_loss"

    if not os.path.exists("saved"):
        os.mkdir("saved")
    if not os.path.exists("saved/" + env_name + "_DQN"):
        os.mkdir("saved/" + env_name + "_DQN")
    if not os.path.exists("saved/" + env_name + "_DQN/models"):
        os.mkdir("saved/" + env_name + "_DQN/models")
    model_save_path = "saved/" + env_name + "_DQN/models/" + train_time + ".pth"

    if not os.path.exists("saved/" + env_name + "_DQN/logs"):
        os.mkdir("saved/" + env_name + "_DQN/logs")
    log_path = "saved/" + env_name + "_DQN/logs/log-" + train_time + ".txt"


class NetworkConfig:
    hidden_dim_1 = 512
    hidden_dim_2 = 128

