import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

parser = argparse.ArgumentParser(prog='plotter')
parser.add_argument('filename')
parser.add_argument('var_path')
parser.add_argument('-d', '--dump', action='store_true', default=False)

args = parser.parse_args()

f = h5py.File(args.filename)

if args.dump:
    print(f.keys())
else:
    this = np.array(f[args.var_path])
    s = plt.contourf(this, levels=30)
    plt.colorbar(s)
    plt.suptitle(args.filename + " " + args.var_path)
    plt.show()
