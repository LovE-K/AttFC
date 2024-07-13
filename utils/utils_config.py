import importlib
import os.path as osp


def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)  # 读取配置文件名
    temp_module_name = osp.splitext(temp_config_name)[0]  # 根据配置文件名读取模型名
    # config = importlib.import_module("configs.base")  # 读取base.py中的配置
    # cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    cfg = config.config
    if cfg.output is None:
        cfg.output = osp.join('work_dirs', temp_module_name)
    return cfg
