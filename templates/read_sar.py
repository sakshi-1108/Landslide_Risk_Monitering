import rasterio
import numpy as np

sar_file = r".\satellite_data\S1D_IW_GRDH_1SDV_20260904T121313_20260904T121338_004428_00833E_A899.SAFE\measurement\s1d-iw-grd-vv-20260904t121313-20260904t121338-004428-00833e-001.tiff"

with rasterio.open(sar_file) as src:

    data = src.read(1)

    valid_data = data[data > 0]

    print("SAR file opened successfully!")
    print("Width:", src.width)
    print("Height:", src.height)
    print("Number of bands:", src.count)
    print("Coordinate system:", src.crs)
    print("Data type:", src.dtypes[0])
    print("Resolution:", src.res)

    print("\nSAR Statistics")
    print("----------------")
    print("Minimum:", np.min(valid_data))
    print("Maximum:", np.max(valid_data))
    print("Mean:", np.mean(valid_data))
    print("Median:", np.median(valid_data))