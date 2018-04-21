import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import pickle as pkl
fn = 0
seed = 0
path_fname  = 'problem2/path{}/fin_max_length{}.pkl'.format(fn,seed) 
with open(path_fname,'rb') as f:
    saw = pkl.load(f)
path = saw
print(len(path))
path = Path(path)
fig = plt.figure()
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, facecolor='white')
ax.add_patch(patch)
ax.set_xlim(-1,11)
ax.set_ylim(-1,11)
plt.savefig('figures/fin_{}_path.png'.format(fn))
