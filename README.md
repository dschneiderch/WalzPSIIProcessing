# WalzPSIIProcessing
A pipeline to process multiframe tifs exported from ImagingWin .pim files

You will need to export your .pim files to multiframe .tifs and then drop them in /data/raw_multiframe
They should be labeled with exactly 2 underscores, 2 descriptors, and a date: e.g. experiment1_20190501_drought.tif

To run the pipeline:

```
cd <project directory>
ipython scripts/psII.py
```

To change threshold values and the plant mask, modify `psIImask()` in `src/segmentation/create_masks.py`

To change the filename format of the multiframe tif you will need to change the `import_snapshots()` in `src/data/import_snapshots.py`
