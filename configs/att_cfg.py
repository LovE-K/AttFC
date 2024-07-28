from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)

config.network = "r50"
config.output = ""  # model output path
config.data_root = ""  # train set path
config.val = ""  # test set path

config.anno_files = 'all_list.txt'  # img list files(created by utils\img_to_list.py)
config.num_classes = 93431  # the number of identity in train set
config.num_image = 5179510  # the number of all images in train set

config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.embedding_size = 512
config.queue_size = int((config.num_classes * 0.3) / config.batch_size) * config.batch_size
config.sample_num = 2 + 1
config.num_epoch = 5
config.batch_size = 384
config.warmup_epoch = 0
config.image_size = (112, 112)
config.resume = False
config.save_all_states = False
config.fp16 = True
config.warmup_epoch = 0
config.total_step = config.num_image // config.batch_size * config.num_epoch
config.warmup_step = config.num_image // config.batch_size * config.warmup_epoch
config.steps_per_epoch = config.total_step // config.num_epoch
config.lr = 0.1
config.optimizer = "sgd"
config.momentum = 0.9
config.weight_decay = 5e-4
config.verbose = 16000
config.frequent = 10
config.gradient_acc = 1
config.num_workers = 8
config.seed = 2048
