import ee
import pandas as pd
import os
import csv
from osgeo import ogr
from matplotlib import pyplot as plt

CLOUD_FILTER = 60
CLD_PRB_THRESH = 40
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 100


def maskCloudsS2(image):
    
    # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR#bands

    # first mask based on qa band
    pixel_qa = image.select('QA60')
    cloud_mask = pixel_qa.bitwiseAnd(1024).eq(0).And(pixel_qa.bitwiseAnd(2048).eq(0))
    image = image.mask(cloud_mask)

    # second mask based on scene classification
    scl = image.select('SCL').unmask(0)
    scl_mask = scl.neq(0).And(    # nodata
               scl.neq(3)).And(   # cloud shadows
                 scl.neq(7)).And(   # unclassified
                 scl.neq(8)).And(   # cloud medium prob
                 scl.neq(9)).And(   # cloud high prob
                 scl.neq(10)).And(  # cirrus
                 scl.neq(11))       # snow
                 
    image = image.updateMask(scl_mask)

    return image.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])


def get_s2_sr_cld_col(aoi, start_date, end_date, cloud_filter):

    # https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless

    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


def calcNdviS2(image):
    result = image.normalizedDifference(["B8A", "B4"]) \
                      .rename("ndvi") \
                      .set('system:time_start', image.get('system:time_start'))
    return result


def plot_ts(t, vi, fit = None):

    plt.figure(figsize=(18, 7))
    ts = plt.plot(t, vi, 'ko', fillstyle='none')
    if fit is not None:
        t_fit, vi_fit = fit
        plt.plot(t_fit, vi_fit, 'blue')

    plt.xlabel('Time (days)')
    plt.ylabel('EVI')
    plt.ylim(0, 1)
    plt.show()


def extract_pixels_toDrive(ic, fc, params=None):
    
    # https://developers.google.com/earth-engine/tutorials/community/extract-raster-values-for-points
  
    # Initialize internal params dictionary.
    _params = {
        "reducer": ee.Reducer.mean(),
        "scale": None,
        "crs": None,
        "crsTransform": None,
        "bands": None,
        "bandsRename": None,
        "imgProps": None,
        "imgPropsRename": None,
        "datetimeName": 'datetime',
        "datetimeFormat": 'YYYY-MM-dd HH:mm:ss'
    }

    # Replace initialized params with provided params.
    if params is not None:
        for param in params:
            _params[param] = params[param] or _params[param]


    # Set default parameters based on an image representative.
    imgRep = ic.first()
    nonSystemImgProps = ee.Feature(None) \
        .copyProperties(imgRep).propertyNames()
    
    if _params["bands"] is None:
        _params["bands"] = imgRep.bandNames()
    if _params["bandsRename"] is None:
        _params["bandsRename"] = _params["bands"]
    if _params["imgProps"] is None:
        _params["imgProps"] = nonSystemImgProps
    if _params["imgPropsRename"] is None:
        _params["imgPropsRename"] = _params["imgProps"]

    # Map the reduceRegions function over the image collection.
    def xtract(img):

        # Select bands (optionally rename), set a datetime & timestamp property.
        img = ee.Image(img.select(_params["bands"], _params["bandsRename"])) \
            .set(_params["datetimeName"], img.date().format(_params["datetimeFormat"])) \
            .set('timestamp', img.get('system:time_start'))

        # Define final image property dictionary to set in output features.
        propsFrom = ee.List(_params["imgProps"]).cat(ee.List([_params["datetimeName"], 'timestamp']))
        propsTo = ee.List(_params["imgPropsRename"]).cat(ee.List([_params["datetimeName"], 'timestamp']))
        imgProps = img.toDictionary(propsFrom).rename(propsFrom, propsTo)

        # Subset points that intersect the given image.
        fcSub = fc.filterBounds(img.geometry())

        # Reduce the image by regions.

        rr_args = {"collection": fcSub, 
                   "reducer": _params["reducer"],
                   "scale": _params["scale"],
                   "crs": _params["crs"],
                   "crsTransform": _params["crsTransform"]}

        return img.reduceRegions(**rr_args).map(lambda x: x.set(imgProps).setGeometry(None)) # Add metadata to each feature.

    results = ic.map(xtract).flatten().filter(ee.Filter.notNull(_params["bandsRename"]))

    return results


def extract_point(imgCollection, x, y=None, scale=None, crs=None, crsTransform=None):


    if isinstance(x, ee.geometry.Geometry):
        pts = x
    else:
        pts = ee.Geometry.Point([x, y])

    xtr = imgCollection.getRegion(geometry=pts,
                                   scale=scale,
                                   crs=crs,
                                   crsTransform=crsTransform).getInfo()

    return pd.DataFrame(xtr[1:], columns=xtr[0]).dropna()


def extract_pixels(imgCollection, shp_file, out_csv_file=None, ids=None, id_field="ID", scale=None, crs=None, crsTransform=None):

    if out_csv_file is not None:
        
        if os.path.exists(out_csv_file):
            print('Skipped %s. File exists.' % out_csv_file)
            return

        print('Writing %s' % out_csv_file)
        fcon = open(out_csv_file, 'w', newline='')
        writer = csv.writer(fcon)
        headerWasNotWritten = True

    drvMemV = ogr.GetDriverByName('Memory')
    src = drvMemV.CopyDataSource(ogr.Open(shp_file), '')

    lyr = src.GetLayer()
    feat = lyr.GetNextFeature()

    while feat:

        pid = feat.GetField(id_field)

        if ids is not None:
            if pid not in ids:
                continue

        geom = feat.GetGeometryRef()

        xCoord = geom.GetX()
        yCoord = geom.GetY()
        pts = {'type': 'Point', 'coordinates': [xCoord, yCoord]}

        vals = imgCollection.getRegion(**{'geometry': pts,
                                          'scale': scale,
                                          'crs': crs,
                                          'crsTransform': crsTransform}).getInfo()

        if headerWasNotWritten:
            writer.writerow(['pid'] + vals[0])
            headerWasNotWritten = False

        for i in range(1, len(vals)):
            if not any(v is None for v in vals[i]):
                writer.writerow([pid] + vals[i])

        # Get next feature
        feat = lyr.GetNextFeature()
    fcon.close()
    drvMemV = None
    src = None

def mask_clouds_mod13(image):
    return image.updateMask(image.select("SummaryQA").lte(1))


def getModisVi(startDate, endDate, region):
    
    # MOD13Q1 : 16-day terra
    # MYD13Q1 : 16 day aqua
    modCollection = ee.ImageCollection("MODIS/006/MOD13Q1") \
                      .filterDate(startDate, endDate) \
                      .filterBounds(region) \
                      .select(['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 
                               'sur_refl_b07', 'NDVI', 'EVI', 'DayOfYear', 'SummaryQA'],
                              ['red', 'nir', 'blue', 'mir', 'ndvi', 'evi', 'doy', 'SummaryQA']) \
                      .map(mask_clouds_mod13) \
                      .map(lambda x: x.set("SATELLITE", "MOD13Q1")) \
                      .map(lambda x: x.select('ndvi').divide(10000).addBands(x.select('doy')).set('system:time_start', x.get('system:time_start')))
                    
    mydCollection = ee.ImageCollection("MODIS/006/MYD13Q1") \
                      .filterDate(startDate, endDate) \
                      .filterBounds(region) \
                      .select(['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 
                               'sur_refl_b07', 'NDVI', 'EVI', 'DayOfYear', 'SummaryQA'],
                              ['red', 'nir', 'blue', 'mir', 'ndvi', 'evi', 'doy', 'SummaryQA']) \
                      .map(mask_clouds_mod13) \
                      .map(lambda x: x.set("SATELLITE", "MYD13Q1")) \
                      .map(lambda x: x.select('ndvi').divide(10000).addBands(x.select('doy')).set('system:time_start', x.get('system:time_start')))
                  
    imgCollection = modCollection.merge(mydCollection)
                
    return imgCollection.select(['ndvi', 'doy'])