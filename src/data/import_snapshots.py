# -*- coding: utf-8 -*-
import os
import glob
import re as re
from datetime import datetime, timedelta
import pandas as pd
from src.data import Multi2Singleframes


def import_snapshots(snapshotdir, camera='vis'):
    '''
    Input:
    snapshotdir = directory of .tif files
    camera = the camera which captured the images. 'vis' or 'psii'

    Export .tif into snapshotdir from LemnaBase using format {0}-{3}-{1}-{6}
    '''

    # %% Get metadata from .tifs
    # snapshotdir = 'data/raw_snapshots/psII'

    # first find the multiframe .tif exports from the pim files
    fns = [fn for fn in glob.glob(pathname=os.path.join(snapshotdir,'raw_multiframe','*.tif'))]
    for fn in fns:
        Multi2Singleframes.extract_frames(fn,'data')

    # now find the individual frame files
    fns = []
    for fname in os.listdir(snapshotdir):
        if re.search(r"_[0-9]+.tif", fname):
            fns.append(fname)

    flist = list()
    fn=fns[0]
    for fn in fns:
        f=re.split('[_\\ ]', os.path.splitext(os.path.basename(fn))[0])
        f.append(os.path.join(snapshotdir,fn))
        flist.append(f)

    fdf=pd.DataFrame(flist,columns=['exp','date','anotherdescriptor','imageid','filename'])

    # convert date and time columns to datetime format
    fdf['date'] = pd.to_datetime(fdf.loc[:,'date'])
    fdf['jobdate'] = fdf['date'] #my scripts use job date so id suggest leaving this. i needed to unify my dates when i image overnigh

    # convert image id from string to integer that can be sorted numerically
    fdf['imageid'] = fdf.imageid.astype('uint8')
    fdf = fdf.sort_values(['exp','date','imageid'])

    fdf = fdf.set_index(['exp','date','jobdate'])
    # check for duplicate jobs of the same sample on the same day.  if jobs_removed.csv isnt blank then you shyould investigate!
    #dups = fdf.reset_index('datetime',drop=False).set_index(['imageid'],append=True).index.duplicated(keep='first')
    #dups_to_remove = fdf[dups].drop(columns=['imageid','filename']).reset_index().drop_duplicates()
    #dups_to_remove.to_csv('jobs_removed.csv',sep='\t')
    #

    return fdf
