import geopandas as gpd
import pandas as pd

# load the district mapping we made earlier
brgys = pd.read_csv("data/brgys_with_districts.csv", index_col=0)[["barangay", "district"]]

# load the gpkg
gdf = gpd.read_file("data/barangay_and_cluster_geo.gpkg")

# merge district in
gdf = gdf.merge(brgys[["barangay", "district"]], on="barangay", how="left")

# check for any unmapped
print(gdf[gdf["district"].isna()]["barangay"].tolist())

# save back to gpkg
gdf.to_file("data/barangay_and_cluster_geo_with_districts.gpkg", driver="GPKG")
print("Columns:", gdf.columns.tolist())