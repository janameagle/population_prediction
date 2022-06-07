
library(raster)
library(ggplot2)

# sequence to loop over years
seq = list('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')


# read MODIS lulc data per year and combine classes based on thematic land cover
# and pixel count

for (y in seq) {
im <- raster(paste0("C:/Users/jmaie/Documents/Masterarbeit/Land_cover/MODIS_yearly/land_cover_20", y, ".tif"))
# im
unique(im)

im[im == 0] <- NA # remove 0 values, they are not a class
im[im == 6] <- 7 # closed and open shrubs
im[im == 8] <- 9 # woody savanna and savanna
im[im == 11] <- 17 # permanent wetlands and water

# im[im == 10] <- ? # grassland
# im[im == 12] <- ? # cropland

unique(im)

writeRaster(im, paste0("C:/Users/jmaie/Documents/Masterarbeit/Land_cover/MODIS_yearly_combined_classes/land_cover_20", y, ".tif"))
}


# show the histogram of pixel counts per class
f <- hist(im)
dat <- data.frame(counts= f$counts,breaks = f$mids)
ggplot(dat, aes(x = breaks, y = counts)) + 
  geom_bar(stat = "identity",fill='blue',alpha = 0.8)+
  xlab("count")+ ylab("class value")+
  scale_x_continuous(breaks = seq(0,17,1),  ## without this you will get the same scale
                     labels = seq(0,17,1))    ## as hist (question picture)

# show the actual pixel counts per class
freq(im, value=0)
freq(im)


