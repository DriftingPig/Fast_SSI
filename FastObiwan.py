import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from tractor import *
from astrometry.util.util import wcs_pv2sip_hdr
import fitsio
from astrometry.util.fits import fits_table
from tractor import PsfExModel
from tractor import PixelizedPsfEx
import tractor.ellipses as ellipses

import sys
sys.path.append("/global/homes/h/huikong/legacypipe/py")
from legacypipe.survey import LegacySurveyData
from legacypipe.survey import wcs_for_brick
from tractor.psf import HybridPixelizedPSF, NCircularGaussianPSF
from tractor.basics import NanoMaggies, LinearPhotoCal
from legacypipe.survey import LegacySurveyWcs
import tractor
from astrometry.util.resample import resample_with_wcs
from tractor.galaxy import DevGalaxy, ExpGalaxy
from tractor.sersic import SersicGalaxy
from legacypipe.survey import LegacySersicIndex
from astrometry.util.util import lanczos3_interpolate

class LegacysurveyCCDList(object):
    def __init__(self, brickname, stamp_hw = 31):
        self.bricname = brickname
        #/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/
        self.survey = LegacySurveyData(survey_dir = "/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr9/")
        self.brick = self.survey.get_brick_by_name(brickname)
        self.targetwcs = wcs_for_brick(self.brick, W=3600, H=3600, pixscale = 0.262)
        self.ccd_list = self.survey.ccds_touching_wcs(self.targetwcs, ccdrad=None)
        self.img_dir = "/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr9/images/"
        self.psf_dir = "/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr9/calib/psfex/"
        self.sky_dir = "/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr9/calib/sky/"
        self.imghw = stamp_hw
        
    def read_sim_gal(self, sim):
        self.ra = sim['ra']
        self.dec = sim['dec']
        self.gflux = sim['sim_gflux']
        self.rflux = sim['sim_rflux']
        self.zflux = sim['sim_zflux']
        self.w1flux = 10**(-(sim['sim_w1']-22.5)/2.5)*sim['mw_transmission_w1']
        self.shape_r = sim['sim_rhalf']
        self.e1 = sim['sim_e1']
        self.e2 = sim['sim_e2']
        self.sersic = sim['sim_sersic_n']
        
    def get_source(self, src_idx, position = 'radec'):
        
        brightness =  tractor.NanoMaggies(g=self.gflux[src_idx], r=self.rflux[src_idx], z=self.zflux[src_idx], order=['g', 'r', 'z'])
        shape = ellipses.EllipseE(self.shape_r[src_idx], self.e1[src_idx], self.e2[src_idx])

        if position == 'radec':
            if self.sersic[src_idx] == 1:
                  self.source_i = ExpGalaxy(RaDecPos(self.ra[src_idx], self.dec[src_idx]), brightness, shape)
            elif self.sersic[src_idx] == 4:
                  self.source_i = DevGalaxy(RaDecPos(self.ra[src_idx], self.dec[src_idx]), brightness, shape)
            elif self.sersic[src_idx] == 0:
                  self.source_i = PointSource(RaDecPos(self.ra[src_idx], self.dec[src_idx]), brightness)
            else:
                  self.source_i = SersicGalaxy(RaDecPos(self.ra[src_idx], self.dec[src_idx]), brightness, shape, LegacySersicIndex(self.sersic[src_idx]))
        elif position == 'pixel': 
            pos = PixPos(self.imghw/2.-1, self.imghw/2.-1)
            if self.sersic[src_idx] == 1:
                  self.source_i = ExpGalaxy(pos, brightness, shape)
            elif self.sersic[src_idx] == 4:
                  self.source_i = DevGalaxy(pos, brightness, shape)
            elif self.sersic[src_idx] == 0:
                  self.source_i = PointSource(pos, brightness)
            else:
                  self.source_i = SersicGalaxy(pos, brightness, shape, LegacySersicIndex(self.sersic[src_idx]))
        else:
            raise ValueError("unrecognized position parameter")
            
        self.src_ra = self.ra[src_idx]
        self.src_dec = self.dec[src_idx]
        self.brightness = brightness
        self.shape = shape
    
        
    def init_one_ccd(self, idx):
        #wcs
        self.ccd_filename = self.ccd_list.image_filename[idx]
        self.ccd_image_hdu = self.ccd_list.image_hdu[idx]
        self.camera = self.ccd_list.camera[idx]
        self.exptime = self.ccd_list.exptime[idx]
        self.ccd_width = self.ccd_list.width[idx]
        self.ccd_height = self.ccd_list.height[idx]
        
        
        header = fitsio.FITS(self.img_dir+self.ccd_filename)[self.ccd_image_hdu].read_header()
        self.raw_twcs = wcs_pv2sip_hdr(header, stepsize=0)
        self.ls_raw_twcs = LegacySurveyWcs(self.raw_twcs, None)
        #self.sig1 = self.ccd_list.sig1[idx]
        self.filter = self.ccd_list.filter[idx]
        im = self.survey.get_image_object(self.ccd_list[idx])
        self.sig1 = im.sig1
        self.ccdzpt = im.ccdzpt
        self.zpscale = NanoMaggies.zeropointToScale(self.ccdzpt)
        self.gain = im.get_gain(primhdr  = None,  hdr = header )
        if self.camera == "decam":
            self.nano2e = self.zpscale*self.gain
        elif self.camera == 'mosaic' or self.camera == '90prime':
            self.nano2e = self.zpscale*self.exptime
        else:
            raise ValueError("camera does not exist")
        
        #psf
        self.psf_fn = self.ccd_filename.replace(".fits.fz", "-psfex.fits")
        T = fits_table(self.psf_dir+self.psf_fn)
        
        header = T.get_header()
        expnum  = self.ccd_list.expnum[ idx ]
        ccdname = self.ccd_list.ccdname[ idx ]
        I, = np.nonzero((T.expnum == expnum) *
                            np.array([c.strip() == ccdname
                                      for c in T.ccdname]))
        assert( len(I) == 1)
        Ti = T[I[0]]
        self.psf_fwhm = Ti.psf_fwhm
        # Remove any padding
        degree = Ti.poldeg1
        # number of terms in polynomial
        ne = (degree + 1) * (degree + 2) // 2
        Ti.psf_mask = Ti.psf_mask[:ne, :Ti.psfaxis1, :Ti.psfaxis2]
        psfex = PsfExModel(Ti=Ti)
        psf = PixelizedPsfEx(None, psfex=psfex)
        self.raw_psf = HybridPixelizedPSF(psf, cx=0, cy=0, gauss=NCircularGaussianPSF([psf.fwhm / 2.35], [1.]))

        #depth: https://github.com/legacysurvey/legacypipe/blob/main/py/legacypipe/coadds.py#L523

    
        
    def set_local(self):
        
        
        flag, xx, yy = self.raw_twcs.radec2pixelxy( self.src_ra, self.src_dec )
        x_cen = xx-1
        y_cen = yy-1
        x_cen_int,y_cen_int = round(x_cen),round(y_cen)
        self.radius = int(self.imghw/2)
        sx0, sx1, sy0, sy1 = x_cen_int-self.radius,x_cen_int+self.radius-1,y_cen_int-self.radius,y_cen_int+self.radius-1
        self.sx0 = sx0
        self.sy0 = sy0
        subslc = slice( sy0, sy1), slice(sx0, sx1)
        subwcs = self.ls_raw_twcs.shifted(sx0, sy0)
        subpsf = self.raw_psf.constantPsfAt((sx0+sx1)/2., (sy0+sy1)/2.)
        photocal=LinearPhotoCal(1, band = self.filter)
        self.tim = Image(np.zeros( (self.imghw,self.imghw)) , invvar=np.ones( (self.imghw,self.imghw) ), wcs = subwcs, psf=subpsf, photocal = photocal )
        self.noise = np.zeros( (self.imghw,self.imghw) )
        self.x_cen_int = x_cen_int
        self.y_cen_int = y_cen_int

        
        
    def gaussian_background(self):
        self.noise = np.random.normal(size=self.tim.shape) * self.sig1
        

    def resample_image(self):
        flag, X, Y = self.targetwcs.radec2pixelxy( self.src_ra, self.src_dec)
        self.target_x = X
        self.target_y = Y
        self.sub_targetwcs = self.targetwcs.get_subimage( int(X)-self.radius, int(Y)-self.radius, self.imghw, self.imghw)

        self.sub_twcs = self.raw_twcs.get_subimage( self.sx0, self.sy0, self.imghw, self.imghw)

        new_tractor = Tractor([self.tim], [self.source_i])
        mod0 = new_tractor.getModelImage(0)+self.noise
        clean_mod0 = new_tractor.getModelImage(0)
        
        X = resample_with_wcs( self.sub_targetwcs, self.sub_twcs, Limages=[mod0], L=3)
        clean_X = resample_with_wcs( self.sub_targetwcs, self.sub_twcs, Limages=[clean_mod0], L=3)

        IMG = np.zeros( (self.imghw, self.imghw) )
        IMG[X[0], X[1]] = X[4][0]

        IMG_clean = np.zeros( (self.imghw, self.imghw) )
        IMG_clean[clean_X[0], clean_X[1]] = clean_X[4][0]

        self.coimg_i = IMG
        self.clean_coimg_i = IMG_clean

        self.ls_sub_targetwcs = LegacySurveyWcs(self.sub_targetwcs,None)
        

    def resample_psf(self):
        #https://github.com/legacysurvey/legacypipe/blob/main/py/legacypipe/coadds.py#L486
        h,w = self.tim.shape
        patch = self.tim.psf.getPointSourcePatch(w//2, h//2).patch
        patch /= np.sum(patch)
        ph,pw = patch.shape
        pscale = LegacySurveyWcs(self.sub_twcs, None).pixscale / self.sub_targetwcs.pixel_scale()
        self.pscale = pscale

        coph = int(np.ceil(ph * pscale))
        copw = int(np.ceil(pw * pscale))
        coph = 2 * (coph//2) + 1
        copw = 2 * (copw//2) + 1
        # want input image pixel coords that change by 1/pscale
        # and are centered on pw//2, ph//2
        cox = np.arange(copw) * 1./pscale
        cox += pw//2 - cox[copw//2]
        coy = np.arange(coph) * 1./pscale
        coy += ph//2 - coy[coph//2]
        fx,fy = np.meshgrid(cox,coy)
        fx = fx.ravel()
        fy = fy.ravel()
        ix = (fx + 0.5).astype(np.int32)
        iy = (fy + 0.5).astype(np.int32)
        dx = (fx - ix).astype(np.float32)
        dy = (fy - iy).astype(np.float32)
        copsf = np.zeros(coph*copw, np.float32)
        lanczos3_interpolate(ix, iy, dx, dy, [copsf], [patch])
        copsf = copsf.reshape((coph,copw))
        copsf /= copsf.sum()
        self.copsf_i = PixelizedPSF(copsf) 
    def finalize_tim(self):
        photocal=LinearPhotoCal(1, band = self.filter)
        noise_bkg = np.ones( (self.imghw, self.imghw) )*self.sig1
        noise_stamp = self.clean_coimg_i/self.nano2e
        noise_stamp[np.where( noise_stamp < 0)] = 0
        assert( np.all(noise_stamp) >=0 )
        noise_sq_tot = noise_bkg**2+noise_stamp
        invvar = 1./noise_sq_tot
        self.final_tim = Image(self.coimg_i , invvar=invvar, wcs = self.ls_sub_targetwcs, psf=self.copsf_i, photocal = photocal )

        patch = self.copsf_i.getPointSourcePatch(16, 16).patch
        self.psfnorm = np.sqrt(np.sum(patch**2))
        self.detsig1 = self.sig1 / self.psfnorm
        self.psfdetiv = 1./self.detsig1**2
    
        

        
        
        
        
        
        

    