import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from itertools import *
import pandas as pd
from collections import defaultdict
from collections import defaultdict

# decemberists (ill be your girl)

# they only look at o-d flows where at least two visitors contribute to the flow
# for each pair of cells

# for each person:
  # for each cell, gather number of times that cell is visited
  # put a 1 in the global (frequency x distance -> people) table
  # for all distances for which there was at least one place the user visited
  # with that frequecy

# The frequency here is about specific places, not distances in general. 
# We want to plot, for each frequency f and radius r, the number of people who
# visit at least one place with distance between r and r+2 of their homes exactly
# f times each month. 

# Verify with Paolo

# Should plot the initial locations
# Should plot aggregate distance traveled per cell
# Can plot normalized measure too:
  # Energy in a cell i is
  # sum over all cells s and frequencies f of dist(s,i) * N_v(s,i,f) * Dout^{-1}(s, f)
  # where D_v(s,i,f) is the number of distinct visitors who reside in the sth cell
  # and visit the ith cell. 

# average distance per visit for each specific month frequency

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
  return np.sqrt(diff.dot(diff)).astype(int)

# use numpy append

def walkers(n):
  # freq_dist_count[i][j] gives the number of people who visit radius j to j+2 with frequency i+1
  freq_dist_count = np.zeros((50,100))
  for i in range(n):
    dist_freq = np.zeros(100).astype(int)
    visits = [1.0]
    S = 1
    home = rnd.uniform(0, 99, 2)
    pos = home
    visited = [pos]
    loc = S
    for _ in range(sample_numhops() + 1):
      dist = distance(home, pos)
      if dist < 100: dist_freq[dist] += 1
      if dist < 101: dist_freq[dist-1] += 1
      if dist < 102: dist_freq[dist-2] += 1
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
    for k,v in enumerate(dist_freq):
      if v > 50: continue
      freq_dist_count[v - 1][k] += 1
  return freq_dist_count

acc_freq_dists = walkers(1000)

plt.figure(figsize=(15,10))
for f in range(50):
  plt.loglog(range(100), acc_freq_dists[f], 'o')
plt.legend(np.arange(50)+1)
plt.savefig('r_vs_people.png', bbox_inches='tight')
plt.clf()

# plt.figure(figsize=(15,10))
# for f in range(10):
#   plt.plot(np.arange(100) * f**2, acc_freq_dists[f], 'o')
# plt.savefig('r_vs_people.png', bbox_inches='tight')
