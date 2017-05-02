import numpy as np


def test_run():
    np.random.seed(693)
    a = np.random.randint(0,10,size=(5,4))
    print('array:', a)

if __name__ == '__main__':
    test_run()