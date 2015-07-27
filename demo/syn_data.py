import random

def syn_data():
    for i in xrange(20000):
        a = random.uniform(0, 10)
        b = random.uniform(0, 10)
        c = a * a + b * b 
        error = random.uniform(-0.01, 0.01)
        print '%.5f\t%.5f\t%.5f' % (a, b, c + error)

if __name__ == '__main__':
    syn_data()

