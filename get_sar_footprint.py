import xml.etree.ElementTree as ET
import os

xml_file = r".\satellite_data\S1D_IW_GRDH_1SDV_20260904T121313_20260904T121338_004428_00833E_A899.SAFE\annotation\s1d-iw-grd-vv-20260904t121313-20260904t121338-004428-00833e-001.xml"

tree = ET.parse(xml_file)
root = tree.getroot()

coordinates = []

# Find latitude and longitude elements
for elem in root.iter():

    if elem.tag.endswith("latitude"):
        lat = elem.text

        if lat:
            coordinates.append(("latitude", float(lat)))

    elif elem.tag.endswith("longitude"):
        lon = elem.text

        if lon:
            coordinates.append(("longitude", float(lon)))


latitudes = [value for name, value in coordinates if name == "latitude"]
longitudes = [value for name, value in coordinates if name == "longitude"]

print("Sentinel-1D SAR footprint")
print("-------------------------")

print("Number of latitude points:", len(latitudes))
print("Number of longitude points:", len(longitudes))

print()

print("Minimum latitude:", min(latitudes))
print("Maximum latitude:", max(latitudes))

print("Minimum longitude:", min(longitudes))
print("Maximum longitude:", max(longitudes))