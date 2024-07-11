#coding=utf-8
import time
from util.conf import ModelConf
from data.loader import FileIO
import warnings
warnings.filterwarnings("ignore")


class DAHNRec(object):
    def __init__(self, config):
        self.config = config
        self.training_data = FileIO.load_data_set(config['training.set'])
        self.test_data = FileIO.load_data_set(config['test.set'])
        self.kwargs = {}
        print('******Reading data and preprocessing...')

    def execute(self):

        import_str = 'from model.' + self.config['model.name'] + ' import ' + self.config['model.name']


        exec(import_str)
        recommender = self.config['model.name'] + '(self.config, self.training_data, self.test_data, **self.kwargs)'

        eval(recommender).execute_recommender()

if __name__ == '__main__':
    for i in range(1):
        model = "DAHNRec"
        s = time.time()

        conf = ModelConf('./conf/' + model + '.conf')

        rec = DAHNRec(conf)
        rec.execute()
        e = time.time()

        print("Running time: %f s" % (e - s))