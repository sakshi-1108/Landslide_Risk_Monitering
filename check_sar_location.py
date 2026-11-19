import rasterio
import os

sar_folder = r".\satellite_data\S1D_IW_GRDH_1SDV_20260904T121313_20260904T121338_004428_00833E_A899.SAFE"

print("Checking Sentinel-1 SAFE structure...")
print()

# Check manifest
manifest = os.path.join(sar_folder, "manifest.safe")

print("Manifest exists:", os.path.exists(manifest))

# Check annotation folder
annotation_folder = os.path.join(sar_folder, "annotation")

print("Annotation folder exists:", os.path.exists(annotation_folder))

if os.path.exists(annotation_folder):

    files = os.listdir(annotation_folder)

    print("\nAnnotation files:")

    for file in files:
        print("-", file)

# Open TIFF and check transform
tiff_file = os.path.join(
    sar_folder,
    "measurement",
    "s1d-iw-grd-vv-20260904t121313-20260904t121338-004428-00833e-001.tiff"
)

print("\nChecking TIFF geolocation...")
print()

with rasterio.open(tiff_file) as src:

    print("CRS:", src.crs)
    print("Transform:", src.transform)
    print("Bounds:", src.bounds)