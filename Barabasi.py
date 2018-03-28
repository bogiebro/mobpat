import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from itertools import *
import pandas as pd
from collections import defaultdict
from collections import defaultdict


alpha = 0.55
h = 100
top = h**alpha
def sample_displacement():
    u = rnd.random()
    return (-((u * (top - 1) - top) / top))**(-1/alpha)


beta = 0.8
h = 17
top2 = h**beta
def sample_delay():
    u = rnd.random()
    return (-((u * (top2 - 1) - top2) / top2))**(-1/beta)


def sample_numhops():
    i = 0
    h = 0
    while h < 24 * 30: 
        i += 1
        h += sample_delay()
    return i


rho = 0.6
gamma = 0.21

def distance(home, pos):
  diff = home - pos
  return np.sqrt(diff.dot(diff))

def walkers(n):
  freq_dist_count = [defaultdict(int) for _ in range(10)]
  for i in range(n):
    dist_freq = defaultdict(int)
    visits = [1.0]
    S = 1
    home = rnd.uniform(0, 99, 2)
    pos = home
    visited = [pos]
    loc = S
    for _ in range(sample_numhops() + 1):
      dist_freq[distance(home, pos)] += 1
      if rnd.random() < rho * S ** -gamma:
        visits.append(1)
        S += 1
        loc = S
        theta = rnd.rand() * 2 * np.pi
        r = sample_displacement()
        h = np.array([np.cos(theta), np.sin(theta)])
        pos = pos + h * r
        visited.append(pos)
      else:
        va = np.array(visits)
        loc = rnd.choice(S, p=va / va.sum())
        visits[loc] += 1
        pos = visited[loc]
    for k,v in dist_freq.items():
      if v > 10: continue
      freq_dist_count[v - 1][k] += 1
  return freq_dist_count

freq_dist_count = walkers(2000000)

plt.figure(figsize=(15,10))

bins = 120
for f in range(10):
    s = pd.Series(freq_dist_count[f])
    bvc, bins = pd.cut(s.index, bins, retbins=True, labels=False)
    vc = s.groupby(bvc).sum()
    plt.loglog(bins[:len(vc)], vc, 'o')
plt.savefig('r_vs_people.png', bbox_inches='tight')
plt.clf()

bins = 120
for f in range(10):
    s = pd.Series(freq_dist_count[f])
    bvc, bins = pd.cut(s.index * (f+1)**2, bins, retbins=True, labels=False)
    vc = s.groupby(bvc).sum()
    plt.loglog(bins[:len(vc)], vc, 'o')
plt.savefig('rf2_vs_people.png', bbox_inches='tight')
