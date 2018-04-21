import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import pickle as pkl
fn = 1
seed = 4669
path_fname  = 'problem2/fn{}/max_length{}.pkl'.format(fn,seed) 
with open(path_fname,'rb') as f:
    path = pkl.load(f)
print(len(path))
path = Path(path)
fig = plt.figure()
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, facecolor='white')
ax.add_patch(patch)
ax.set_xlim(-1,11)
ax.set_ylim(-1,11)
plt.savefig('figures/{}_path.png'.format(fn))
