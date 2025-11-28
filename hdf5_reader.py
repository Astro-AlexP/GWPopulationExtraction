import numpy as np
import glob
import pandas as pd

pe_dir = "./Data/GW*.hdf5"
ifar_thr = 1.0
nsel = 3000
pe_files = np.sort(glob.glob(pe_dir))

pe, i = {}, 0
for i, ff in enumerate(pe_files):
    with pd.HDFStore(ff, mode='r') as store:
        super_event = ff.rsplit("/")[-1][:-5]
        #ifar = [int(s) for s in super_event.split('_') if s.isdigit()][-1]
        #if ifar < ifar_thr:
        #    continue
        pe[super_event] = {}

        for key in ['mass_1', 'm1']:
            try:
                pe[super_event]['mass1'] = store.get(key).values
            except:
                pass
        for key in ['mass_2', 'm2']:
            try:
                pe[super_event]['mass2'] = store.get(key).values
            except:
                pass
        for key in ['mass_1_source', 'm1_source']:
            try:
                pe[super_event]['mass1_src'] = store.get(key).values
            except:
                pass
        for key in ['mass_2_source', 'm2_source']:
            try:
                pe[super_event]['mass2_src'] = store.get(key).values
            except:
                pass
        for key in ['spin_1z', 'a1z']:
            try:
                pe[super_event]['spin1z'] = store.get(key).values
            except:
                pass
        for key in ['spin_2z', 'a2z']:
            try:
                pe[super_event]['spin2z'] = store.get(key).values
            except:
                pass
        for key in ['luminosity_distance', 'dist']:
            try:
                pe[super_event]['lumd'] = store.get(key).values
            except:
                pass

        print(pe[super_event].keys())


