import sys
sys.path.append("./")
from FastObiwan import * 


#copied from https://github.com/legacysurvey/legacypipe/blob/main/py/legacypipe/runbrick.py#L2610C1-L2674C38
def get_fiber_fluxes(cat, targetwcs, H=32, W=32, pixscale=0.262, bands=['g','r','z'],
                     fibersize=1.5, seeing=1., year=2020.0,
                     plots=False, ps=None):
    from tractor import GaussianMixturePSF
    from legacypipe.survey import LegacySurveyWcs
    import astropy.time
    from tractor.tractortime import TAITime
    from tractor.image import Image
    from tractor.basics import LinearPhotoCal
    from photutils.aperture import CircularAperture, aperture_photometry

    # Create a fake tim for each band to construct the models in 1" seeing
    # For Gaia stars, we need to give a time for evaluating the models.
    mjd_tai = astropy.time.Time(year, format='jyear').tai.mjd
    tai = TAITime(None, mjd=mjd_tai)
    # 1" FWHM -> pixels FWHM -> pixels sigma -> pixels variance
    v = ((seeing / pixscale) / 2.35)**2
    data = np.zeros((H,W), np.float32)
    inverr = np.ones((H,W), np.float32)
    psf = GaussianMixturePSF(1., 0., 0., v, v, 0.)
    wcs = LegacySurveyWcs(targetwcs, tai)
    faketim = Image(data=data, inverr=inverr, psf=psf,
                    wcs=wcs, photocal=LinearPhotoCal(1., bands[0]))

    # A model image (containing all sources) for each band
    modimgs = [np.zeros((H,W), np.float32) for b in bands]
    # A blank image that we'll use for rendering the flux from a single model
    onemod = data

    # Results go here!
    fiberflux    = np.zeros((len(cat),len(bands)), np.float32)
    fibertotflux = np.zeros((len(cat),len(bands)), np.float32)

    # Fiber diameter in arcsec -> radius in pix
    fiberrad = (fibersize / pixscale) / 2.

    # For each source, compute and measure its model, and accumulate
    for isrc,src in enumerate(cat):
        if src is None:
            continue
        # This works even if bands[0] has zero flux (or no overlapping
        # images)
        ums = src.getUnitFluxModelPatches(faketim)
        assert(len(ums) == 1)
        patch = ums[0]
        if patch is None:
            continue
        br = src.getBrightness()
        for iband,(modimg,band) in enumerate(zip(modimgs,bands)):
            flux = br.getFlux(band)
            #flux_iv = T.flux_ivar[isrc, iband]
            #if flux <= 0 or flux_iv <= 0:
            #    continue
            # Accumulate into image containing all models
            patch.addTo(modimg, scale=flux)
            # Add to blank image & photometer
            patch.addTo(onemod, scale=flux)
            sx,sy = faketim.getWcs().positionToPixel(src.getPosition())
            aper = CircularAperture((sx, sy), fiberrad)
            p = aperture_photometry(onemod, aper)
            f = p.field('aperture_sum')[0]
            if not np.isfinite(f):
                # If the source is off the brick (eg, ref sources), can be NaN
                continue
            fiberflux[isrc,iband] = f
            print(f)


def write_cat(catalog):
    ra = []
    dec = []
    shape_r = []
    e1 = []
    e2 = []
    sersic = []
    flux_g = []
    flux_r = []
    flux_z = []
    
    
    for i in range( len(catalog) ):
        ra.append( catalog[i].pos.ra )
        dec.append( catalog[i].pos.dec )
        flux_g.append( catalog[i].brightness.g )
        flux_r.append( catalog[i].brightness.r )
        flux_z.append( catalog[i].brightness.z )
        
        if catalog[i].getSourceType() == 'SersicGalaxy':
            sersic.append( catalog[i].sersicindex.val )
            shape_r.append( catalog[i].shape.re )
            e1.append( catalog[i].shape.e1 )
            e2.append( catalog[i].shape.e2 )
            
        if catalog[i].getSourceType() == 'ExpGalaxy':
            sersic.append( 1 )
            shape_r.append( catalog[i].shape.re )
            e1.append( catalog[i].shape.e1 )
            e2.append( catalog[i].shape.e2 )
            
        if catalog[i].getSourceType() == 'DevGalaxy':
            sersic.append( 4 )
            shape_r.append( catalog[i].shape.re )
            e1.append( catalog[i].shape.e1 )
            e2.append( catalog[i].shape.e2 )
            
        if catalog[i].getSourceType() == 'RexGalaxy':
            sersic.append( 1 )
            shape_r.append( catalog[i].shape.re )
            e1.append( 0 )
            e2.append( 0 )
            
        if catalog[i].getSourceType() == 'PointSource':
            sersic.append( 0 )
            shape_r.append( 0 )
            e1.append( 0 )
            e2.append( 0 )
        from astrometry.util.fits import fits_table
        T = fits_table()
        T.ra = np.array(ra)
        T.dec = np.array(dec)
        T.flux_g = np.array(flux_g)
        T.flux_r = np.array(flux_r)
        T.flux_z = np.array(flux_z)
        T.e1 = np.array(e1)
        T.e2 = np.array(e2)
        T.shape_r = np.array(shape_r)
        T.sersic = np.array(sersic)
        T.writeto("test2.fits")
        

def fit_one_brick(brickname, input_catalog):
    
    input_catalog = input_catalog[input_catalog['brickname'] == brickname]
    Obiwan_Fast = LegacysurveyCCDList(brickname)
    Obiwan_Fast.read_sim_gal( input_catalog ) 

    L_sim = len( Obiwan_Fast.ra )

    catalog = []
    #loop here
    print(L_sim)
    #should be L_sim
    for i in range(L_sim):
        print( i )
        Obiwan_Fast.get_source(i)
   
        L_CCDs = len( Obiwan_Fast.ccd_list)
        tims = []
        for j in range(L_CCDs):
            if Obiwan_Fast.ccd_list.camera[j] != 'decam':
                  continue
            Obiwan_Fast.init_one_ccd(j)
            Obiwan_Fast.set_local()
            if Obiwan_Fast.x_cen_int<10 or Obiwan_Fast.x_cen_int>=Obiwan_Fast.ccd_width-10 or Obiwan_Fast.y_cen_int<10 or Obiwan_Fast.y_cen_int>=Obiwan_Fast.ccd_height-10:
                 #this source does not overlap this ccd
                 continue
            
            Obiwan_Fast.gaussian_background()
            Obiwan_Fast.resample_image()
            Obiwan_Fast.resample_psf()
            Obiwan_Fast.finalize_tim()
            tim_j = Obiwan_Fast.final_tim
            tims.append(tim_j)
            #print(Obiwan_Fast.gain, Obiwan_Fast.zpscale, Obiwan_Fast.sig1,Obiwan_Fast.nano2e)
            #make a guess src that deviates from the true flux
        guess_src = Obiwan_Fast.source_i.copy()
        guess_src.brightness.g = Obiwan_Fast.source_i.brightness.g*(1+np.random.normal()*0.3)
        guess_src.brightness.r = Obiwan_Fast.source_i.brightness.r*(1+np.random.normal()*0.3)
        guess_src.brightness.z = Obiwan_Fast.source_i.brightness.z*(1+np.random.normal()*0.3)
        if Obiwan_Fast.source_i.getSourceType() == 'SersicGalaxy':
            if Obiwan_Fast.source_i.sersicindex.val < 0.3:
                guess_src.sersicindex = LegacySersicIndex(0.3)
            if Obiwan_Fast.source_i.sersicindex.val > 5.3:
                guess_src.sersicindex = LegacySersicIndex(5.3)
        new_tractor = Tractor( tims, [guess_src])
        new_tractor.freezeParam('images')
        for i in range(50):
            dlnp,X,alpha = new_tractor.optimize()
            #print('dlnp', dlnp)
            if dlnp < 1e-3:
                break
        catalog.append( new_tractor.catalog[0] )
    write_cat(catalog)

if __name__ == "__main__":
    Obiwan_LRG = fitsio.read("/global/cfs/cdirs/desi/survey/catalogs/image_simulations/LRG/NGC/Obiwan_LRGs.fits", \
                         columns = ["brickname", "ra", "dec", "sim_gflux", "sim_rflux", "sim_zflux", \
                                    "sim_w1", "mw_transmission_w1", "sim_rhalf", "sim_e1", "sim_e2","sim_sersic_n"])
    brickname = '2383p305'
    
    fit_one_brick(brickname, Obiwan_LRG)


