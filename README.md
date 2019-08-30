# WalzPSIIProcessing
A pipeline to process multiframe tifs exported from ImagingWin .pim files

You will need to export your .pim files to multiframe .tifs and then drop them in `data/raw_multiframe`

They should be labeled with exactly 2 dashes, 2 descriptors, and a date: e.g. drought-20190501-sample1.tif

Additionally you will need `pimframes_map.csv` to describe each frame and `genotype_map.csv` to describe the genotype of each plant. See `example_data/` for guidance.

It is assumed the first descriptor is the treatment, the second descriptor is a sample number.

Additionally you will need pimframes_map.csv to describe each frame and genotype_map.csv to describe the genotype of each plant. It is important that metadata of your filename descriptors and your images match.

The contents of `data/genotype_map.csv` should have 3 column headers EXACTLY as specified here:
```
treatment,roi,gtype
drought, 0, WT
drought, 1, m1
drought, 2, m1
drought, 3, WT
```
The contents of `data/pimframes.csv` should have 3 column headers EXACTLY as specified here:

```
imageid,frame,parameter
1,Fo,FvFm
2,Fm,FvFm
3,AbsR,AbsR
4,AbsN,AbsN
5,Fp,FvFm
6,Fmp,FvFm
7,Fp,t40
8,Fmp,t40
```

See `example_data/` for complete examples and guidance.

To run the pipeline, make sure your plantcv python environment is in your path, then:

```
cd <project directory>
ipython scripts/psII.py
```

To change threshold values and the plant mask, modify `psIImask()` in `src/segmentation/create_masks.py`

At the very least, you will need need to modify `scripts/psII.py` in 2 places:
1. you must change `indir` to point to your data directory.
2. you must change the location of your roi's, currently on line 96 of `scripts/psII.py`. See [plantcv documentation](!https://plantcv.readthedocs.io/en/stable/roi_multi/) for details.