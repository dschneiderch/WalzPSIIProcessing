# %% Setup
# Export .tif into outdir from LemnaBase using format {0}-{3}-{1}-{6}
import os
import re as re
from plantcv import plantcv as pcv
import cv2 as cv2
from collections import defaultdict
import pandas as pd
import glob
import numpy as np
from datetime import datetime, timedelta
import importlib
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
from matplotlib import pyplot as plt

# setup data explorer in atom
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None

# setup plotting
# use inline or agg for below the text, or in atom it can be in the panel sidebar using the settings
# use auto for a new window
#%matplotlib auto
# matplotlib.use('Agg')

# %% io directories
indir = os.path.join('data')
# snapshotdir = indir
outdir = os.path.join('output','psII')
if not os.path.exists(outdir):
    os.makedirs(os.path.join(outdir,'pseudocolor_images'))


# %% Import tif files
from src.data import import_snapshots
#importlib.reload(import_snapshots)
fdf = import_snapshots.import_snapshots(indir, 'psii')

# %% Define the frames from the PSII measurements
fdf.loc[fdf.imageid == 1,'frame'] = 'Fo'
fdf.loc[fdf.imageid == 2,'frame'] = 'Fm'
fdf.loc[fdf.imageid == 3,'frame'] = 'AbsR'
fdf.loc[fdf.imageid == 4,'frame'] = 'AbsN'
fdf.loc[fdf.imageid.isin(np.arange(5,40,2)),'frame'] = 'Fp'
fdf.loc[fdf.imageid.isin(np.arange(6,41,2)),'frame'] = 'Fmp'

# %% Define the parmaeters to be derived from the frames. Generally each measurement comes as a pair of images
# Dark measurements
fdf_dark=fdf.copy()
# fdf.query('experiment.str.contains("dark")', engine='python') #query syntax for partial stringmatch on columns

fdf_dark.loc[fdf_dark.imageid == 1,'parameter']='FvFm'
fdf_dark.loc[fdf_dark.imageid == 2,'parameter']='FvFm'
fdf_dark.loc[fdf_dark.imageid == 3,'parameter']='AbsR'
fdf_dark.loc[fdf_dark.imageid == 4,'parameter']='AbsN'
fdf_dark.loc[fdf_dark.imageid == 5,'parameter']='FvFm'
fdf_dark.loc[fdf_dark.imageid == 6,'parameter']='FvFm'
fdf_dark.loc[fdf_dark.imageid == 7,'parameter']='Y1_ALon'
fdf_dark.loc[fdf_dark.imageid == 8,'parameter']='Y1_ALon'
fdf_dark.loc[fdf_dark.imageid == 9,'parameter']='Y2_ALon'
fdf_dark.loc[fdf_dark.imageid == 10,'parameter']='Y2_ALon'
fdf_dark.loc[fdf_dark.imageid == 11,'parameter']='Y3_ALon'
fdf_dark.loc[fdf_dark.imageid == 12,'parameter']='Y3_ALon'
fdf_dark.loc[fdf_dark.imageid == 13,'parameter']='Y4_ALon'
fdf_dark.loc[fdf_dark.imageid == 14,'parameter']='Y4_ALon'
fdf_dark.loc[fdf_dark.imageid == 15,'parameter']='Y5_ALon'
fdf_dark.loc[fdf_dark.imageid == 16,'parameter']='Y5_ALon'


# %% remove absorptivity measurements which are blank images
df = (fdf_dark
        .query('~parameter.str.contains("Abs") and ~parameter.str.contains("FRon")', engine='python')
        )

# %% remove the duplicate Fm and Fo frames where frame = Fmp and Fp from imageid 5,6
df = df.query('(parameter!="FvFm") or (parameter=="FvFm" and (frame=="Fo" or frame=="Fm") )')


# %% Import function to make mask and setup image classifications
from src.segmentation import createmasks
#importlib.reload(createmasks)  #you can use this to reload the module when you're iterating
from src.viz import custom_colormaps
#importlib.reload(custom_colormaps)
# %% image fucntion

def image_avg(fundf):

    fn_min = fundf.filename.iloc[0]
    fn_max = fundf.filename.iloc[1]
    param_name = fundf['parameter'].iloc[0]

    outfn = os.path.splitext(os.path.basename(fn_max))[0]
    outfn_split = outfn.split('_')
    outfn_split[-1] = param_name
    outfn = "_".join(outfn_split)
    print(outfn)

    sampleid = outfn_split[0]

    if pcv.params.debug == 'print':
        debug_outdir = os.path.join(outdir,'debug',outfn)
        if not os.path.exists(debug_outdir):
            os.makedirs(debug_outdir)
        pcv.params.debug_outdir = debug_outdir

    #read images and create mask from max fluorescence
    imgmax,_,_ = pcv.readimage(fn_max)
    imgmin,_,_ = pcv.readimage(fn_min) #read image as is. only gray values in PSII images
    mask = createmasks.psIImask(imgmax)
    img=imgmax #use img for the boilerplate functions below
     
    c,h = pcv.find_objects(img, mask)
    roi_c,roi_h=pcv.roi.multi(img, coord=(240,180), radius=30, spacing = (160,150), ncols=2, nrows=2)

#    # Copy mask and convert 0,1 to use as an output template later
#    mask2 = mask.copy()
#    mask2[mask2>0]=1
 
    # Make as many copies of incoming dataframe as there are ROIs
    outdf = fundf.copy()
    for i in range(0,len(roi_c)-1):
        outdf = outdf.append(fundf)
    outdf.imageid = outdf.imageid.astype('uint8')

    # Initialize lists to store variables for each ROI and iterate
    frame_avg = []
    psIIparam_avg = []
    psIIparam_std = []
    ithroi = []
    i=1
    rc=roi_c[i]
    for i,rc in enumerate(roi_c):
        # Store iteration Number
        ithroi.append(int(i))
        ithroi.append(int(i)) #append twice so each image has a value.
        # extract ith hierarchy
        rh = roi_h[i]

        # Filter objects based on being in the ROI
        roi_obj, hierarchy_obj, submask, obj_area  = pcv.roi_objects(img, 'partial', rc, rh, c, h)

        if len(roi_obj)==0:

            frame_avg.append(np.nan)
            frame_avg.append(np.nan)
            psIIparam_avg.append(np.nan)
            psIIparam_avg.append(np.nan)
            psIIparam_std.append(np.nan)
            psIIparam_std.append(np.nan)

        else:

            # Combine multiple plant objects within an roi together
            plant_contour, plant_mask = pcv.object_composition(img=img, contours=roi_obj, hierarchy=hierarchy_obj)

            imgmax_masked = np.ma.array(imgmax, mask=~plant_mask.astype('bool'))
            imgmin_masked = np.ma.array(imgmin, mask=~plant_mask.astype('bool'))

            Fdiff = (imgmax_masked - imgmin_masked)
            YII = np.divide(Fdiff, imgmax_masked, where = imgmax_masked!=0)

            frame_avg.append(imgmin_masked.mean())
            frame_avg.append(imgmax_masked.mean())
            psIIparam_avg.append(YII.mean()) #need double because there are two images
            psIIparam_avg.append(YII.mean())
            psIIparam_std.append(YII.std())
            psIIparam_std.append(YII.std())

    
    # Print false color image of all plants together
    imgmax_masked = np.ma.array(imgmax, mask=~mask.astype('bool'))
    imgmin_masked = np.ma.array(imgmin, mask=~mask.astype('bool'))

    Fdiff = (imgmax_masked - imgmin_masked)
    YII = np.divide(Fdiff, imgmax_masked, where = imgmax_masked!=0)
    mycmap = custom_colormaps.get_cmap('imagingwin')
    pseudoimgdir = os.path.join(outdir,'pseudocolor_images',sampleid)
    os.makedirs(pseudoimgdir,exist_ok=True)
    yii_img = pcv.visualize.pseudocolor(YII, mask=mask, cmap=mycmap, min_value = 0, max_value = 1, background='black')
    yii_img.savefig(os.path.join(pseudoimgdir, outfn + '_YII.png'), bbox_inches='tight')
    yii_img.clf()

    # Save statistics from each plant
    outdf['roi'] = ithroi
    outdf['frame_avg'] = frame_avg
    outdf['param_avg'] =  psIIparam_avg
    outdf['param_std'] = psIIparam_std

    return(outdf)

# %% Setup output
# %matplotlib inline # comment this if running script from cmd
pcv.params.debug = 'print' #'print' #'plot', 'print'
plt.rcParams["figure.figsize"] = (12,12)

# %% Compute image average and std for min/max fluorescence
df = df.sort_values(['exp','date','imageid']) #must group so there are pair of images Fp and Fmp or Fo and Fm. make sure df was sorted by datetime and imageid at least
param_order = df.parameter.unique() #this only works if every category is represented in the first day in the dataframe
df['parameter'] = pd.Categorical(df.parameter,categories=param_order, ordered=True)

# del fundf
dfreset = (df#.query('(jobdate=="2019-04-09" | jobdate=="2019-04-16") and (sampleid=="B4")')#.query('(jobdate=="2018-12-15" or jobdate=="2018-12-14") and (sampleid=="a2" or sampleid=="a3")')
  # .reset_index()
 )
fundf=dfreset.iloc[[0,1]]
fundf
del fundf

# %% groupby loop works better with multiple ROI
dfgrps = dfreset.groupby(['exp','jobdate','date','parameter'])
grplist=[]
for grp,grpdf in dfgrps:
    grplist.append(image_avg(grpdf))
df_avg = pd.concat(grplist)

## Add genotype information
gtypeinfo = pd.read_csv('data/genotype_layout.csv')
df_avg = (pd.merge(df_avg.reset_index(),
                  gtypeinfo,
                  on=['exp','roi'],
                  how='outer')
                  )

(df_avg.sort_values(['exp','date','imageid'])
       .to_csv(os.path.join(outdir,'exp1_psII.csv'), na_rep='nan',float_format='%.4f', index=False)
)
