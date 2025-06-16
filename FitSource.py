import sys
sys.path.append("./")
from FastObiwan import * 


#copied from https://github.com/legacysurvey/legacypipe/blob/main/py/legacypipe/runbrick.py#L2610C1-L2674C38
def get_fiber_fluxes(cat, targetwcs, H=64, W=64, pixscale=0.262, bands=['g','r','z'],
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

Obiwan_LRG = fitsio.read("/global/cfs/cdirs/desi/survey/catalogs/image_simulations/LRG/NGC/Obiwan_LRGs.fits", \
                         columns = ["brickname", "sim_ra", "sim_dec", "sim_gflux", "sim_rflux", "sim_zflux", "sim_w1", "mw_transmission_w1", "sim_rhalf", "sim_e1", "sim_e2","sim_sersic_n"])

#loop here
brickname = '2383p305'
Obiwan_LRG = Obiwan_LRG[Obiwan_LRG['brickname'] == brickname]

Obiwan_Fast = LegacysurveyCCDList(brickname)
Obiwan_Fast.read_sim_gal( Obiwan_LRG )
#loop here
Obiwan_Fast.get_source(0)
#loop here
L_CCDs = len( Obiwan_Fast.ccd_list)

tims = []
for i in range(L_CCDs):
    Obiwan_Fast.init_one_ccd(i)
    Obiwan_Fast.set_local()
    Obiwan_Fast.gaussian_background()
    Obiwan_Fast.resample_image()
    Obiwan_Fast.resample_psf()
    Obiwan_Fast.finalize_tim()
    tim_i = Obiwan_Fast.final_tim
    tims.append(tim_i)
    #print(Obiwan_Fast.gain, Obiwan_Fast.zpscale, Obiwan_Fast.sig1,Obiwan_Fast.nano2e)

Obiwan_Fast.get_source(0)
Obiwan_Fast.source_i.brightness.g = Obiwan_Fast.source_i.brightness.g*(1+np.random.normal()*0.03)
Obiwan_Fast.source_i.brightness.r = Obiwan_Fast.source_i.brightness.r*(1+np.random.normal()*0.03)
Obiwan_Fast.source_i.brightness.z = Obiwan_Fast.source_i.brightness.z*(1+np.random.normal()*0.03)


