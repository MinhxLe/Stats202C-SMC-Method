import pandas as pd
import numpy as np
import argparse
import os
import glob
parser = argparse.ArgumentParser()
parser.add_argument('--fn', help='number of samples', type=int)
parser.add_argument('--fin', help='n to n',type=bool)
args = parser.parse_args()

if args.fin:
    fin_str = "fin_"
else:
    fin_str = ""
data = None
checked = False
max_len=int(1e7)
for fname in glob.glob("./problem2/{}fn{}/*.csv".format(fin_str,args.fn)):
    if os.stat(fname).st_size == 0:
        continue
    fdata = pd.read_csv(fname).values
    if not checked and  data == None:
        data = fdata
        checked = True
    else:
        data = np.append(data,fdata,axis=0)
    if len(data) > max_len:
        break
print(len(data))


