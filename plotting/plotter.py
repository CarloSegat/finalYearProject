import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.stats

from utils import load_gzip

def plot(data, name):
    kde = scipy.stats.gaussian_kde(data,bw_method=None)
    t_range = np.linspace(-2,8,200)
    plt.plot(t_range,kde(t_range),lw=2)
    plt.xlim(0,1)
    plt.legend(loc='best')
    #plt.show()

data = load_gzip('FF-ACD-Results')

embs = ['komn', 'google', 'yelp']
synt = ['synt', 'no-synt']
stop = ['stop-kept', 'stop-removed']
punct = ['punct-kept', 'punct-removed']

for e in embs:
    for s in synt:
        for st in stop:
            for p in punct:
                plot(data[e][s][st][p], e+ ' '+s+ ' '+st+ ' '+p)
    plt.savefig(e + '.png')
    plt.clf()
    plt.cla()
    plt.close()

all_scores = {'komn':{'synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                     'stop-removed':{'punct-kept':[], 'punct-removed':[]}},
             'no-synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                        'stop-removed':{'punct-kept':[], 'punct-removed':[]}}
            },

             'google':{'synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                             'stop-removed':{'punct-kept':[], 'punct-removed':[]}},
                     'no-synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                                'stop-removed':{'punct-kept':[], 'punct-removed':[]}}
                    },

             'glove':{'synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                             'stop-removed':{'punct-kept':[], 'punct-removed':[]}},
                     'no-synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                                'stop-removed':{'punct-kept':[], 'punct-removed':[]}}
                    },
            'yelp':{'synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                             'stop-removed':{'punct-kept':[], 'punct-removed':[]}},
                     'no-synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                                'stop-removed':{'punct-kept':[], 'punct-removed':[]}}
                    }
            }