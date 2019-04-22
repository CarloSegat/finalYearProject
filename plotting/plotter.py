import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.stats, sys
sys.path.append('..\.')

from utils import load_gzip

embs = ['komn', 'google', 'yelp']
synt = ['synt', 'no-synt']
stop = ['stop-kept', 'stop-removed']
punct = ['punct-kept', 'punct-removed']


def get_syntax_and_noSyntax_mean(data, emb):
    syntax, no_syntax = 0, 0
    for e in embs:
        for s in synt:
            for st in stop:
                for p in punct:
                    if e == emb:
                        if s == 'no-synt':
                            no_syntax += sum(data[e][s][st][p])
                        else:
                            syntax += sum(data[e][s][st][p])
    return format(syntax / 4*10, '.4f') + '  ' +  format(no_syntax / 4*10, '.4f')

def plot(data, name):
    kde = scipy.stats.gaussian_kde(data,bw_method=None)
    t_range = np.linspace(-2,8,200)

    plt.plot(t_range,kde(t_range),lw=2, label=name)
    plt.xlim(0,1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    #plt.show()

data_ff_acd = load_gzip('FF-ACD-Results')
data_cnn_acd = load_gzip('CNN-ACD-Results')

data = data_ff_acd

new_legend = {'stop-kept': 'stopwords', 
              'stop-removed': 'no-stopwprds',
              'punct-kept': 'punctuation',
              'punct-removed': 'no-punctuation',
              'synt': 'syntax', 'no-synt': 'no-syntax'}

for e in embs:
    for s in synt:
        for st in stop:
            for p in punct:
                legend = e+ ' '+new_legend[s]+ ' '+new_legend[st]+ ' '+new_legend[p]
                plot(data[e][s][st][p], legend)
    #plt.tight_layout()
    plt.figtext(.15, .80, 'F1 mean for syntax and no-syntax:\n' \
    + get_syntax_and_noSyntax_mean(data, e))
    plt.savefig('img/'+ e + '.png', figsize=(100,50), bbox_inches='tight')
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
