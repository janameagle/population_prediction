
# this script is for reading, resampling and stacking the data, 
# data will be used in the ViT model

library(raster)
path <- "C:/Users/jmaie/Documents/Masterarbeit/"
library(rgdal)
study_area <- readOGR(dsn = paste0(path, "study_area_small.gpkg"))

# sequence to loop over years
seq = list('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')
# y = '01'


# read all input data
## static features
# slope from Copernicus DEM
slope <- raster(paste0(path, "DEM/Slope_PSAD_deg_repr_epsg4326.tif"))
# Distance to streets from OSM
streets_dist <- raster(paste0(path, "Distance_grids/highway_distances.tif"))
# Distance to water from Copernicus DEM waterbodies and OSM waterways
water_dist <- raster(paste0(path, "Distance_grids/water_distances.tif"))
# Distance to city center
center_dist <- raster(paste0(path, "Distance_grids/city_center_distances.tif"))


# resample
slope_r <- resample(slope, pop, method='ngb')
streets_dist_r <- resample(streets_dist, pop, method='ngb')
water_dist_r <- resample(water_dist, pop, method='ngb')
center_dist_r <- resample(center_dist, pop, method='ngb')


for (y in seq){
print(y)
## multitemporal features
# WorldPop population grid
pop <- raster(paste0(path, "Population/yearly_pop/20",
                     y, "_UNadj_clipped.tif"))
# Distance to urban extent (derived from population grid)
urb_dist <- raster(paste0(path, "Distance_grids/Urban_extent_distances/20",
                          y, "_urban_ext_pop10_distances.tif"))
# Modis Land Cover
lc <- raster(paste0(path, "Land_cover/MODIS_yearly/land_cover_20", y, ".tif"))


# resample to same resolution
urb_dist_r <- resample(urb_dist, pop, method='ngb')
lc_r <- resample(lc, pop, method='ngb')

# create one raster brick
b <- brick(list(pop, urb_dist_r, lc_r, slope_r, streets_dist_r, water_dist_r, center_dist_r))


# crop to small study area
b_cropped <- crop(b,study_area)



# save raster brick
file <- writeRaster(b_cropped, filename=paste0(path, 'Code/population_prediction/data/brick_20', y, '.tif'), 
            format="GTiff", overwrite=FALSE)

}

