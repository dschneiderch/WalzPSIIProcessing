# WalzPSIIProcessing
A pipeline to process multiframe tifs exported from ImagingWin .pim files

You will need to export your .pim files to multiframe .tifs and then drop them in /data/raw_multiframe

They should be labeled with exactly 2 dashes, 2 descriptors, and a date: e.g. experiment1-20190501-ds.tif

Additionally you will need `pimframes_map.csv` to describe each frame and `genotype_map.csv` to describe the genotype of each plant. See `example_data/` for guidance.

To run the pipeline:

```
cd <project directory>
ipython scripts/psII.py
```

To change threshold values and the plant mask, modify `psIImask()` in `src/segmentation/create_masks.py`

To change the filename format of the multiframe tif you will need to change the `import_snapshots()` in `src/data/import_snapshots.py`
