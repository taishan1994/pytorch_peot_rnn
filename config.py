class Config:
    def __init__(self):
        self.data_dir = './data/'
        self.data_path = self.data_dir + 'peot.txt'
        self.pickle_path = self.data_dir + 'tang.npz'
        self.load_path = './checkpoints/peot.pt'
        self.save_path = './checkpoints/peot.pt'

        self.do_train = False
        self.do_test = False
        self.do_predict = True
        self.do_load_model = True

        self.num_epoch = 20
        self.batch_size = 128
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.max_gen_len = 200
        self.max_len = 125
        self.embedding_dim = 300
        self.hidden_dim = 256
