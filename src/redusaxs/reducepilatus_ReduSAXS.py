import os
import re
import numpy as np
import fabio
import matplotlib.pyplot as plt
from pyFAI import AzimuthalIntegrator
from pyFAI.detectors import Detector
import math
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
from matplotlib.patches import Wedge


def _common_label_prefix(labels):
    """Longest prefix shared by all labels, cut after a '-' or '_' so no name is split mid-token."""
    if len(labels) < 2:
        return ''
    prefix = os.path.commonprefix(labels)
    cut = max(prefix.rfind('-'), prefix.rfind('_'))
    return prefix[:cut + 1] if cut >= 3 else ''


def place_legend_outside(fig, ax, max_width_frac=0.55):
    """
    Legend to the right of the axes, split into as many columns as needed to
    fit the axes height. Kept out of tight_layout, which otherwise squashes
    the axes vertically as soon as the legend is taller than the plot.
    """
    # A canvas on a hidden tab reports a zero-size figure, and laying out
    # against it makes tight_layout invert a singular transform.
    if fig.bbox.width < 1 or fig.bbox.height < 1:
        return
    old = ax.get_legend()
    if old is not None:
        old.remove()
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        fig.tight_layout()
        return

    prefix = _common_label_prefix(labels)
    short_labels = [label[len(prefix):] for label in labels]
    title = prefix.rstrip('-_') or None

    renderer = fig.canvas.get_renderer()
    fig_w = fig.bbox.width
    for fontsize in (8, 7, 6):
        fig.tight_layout()
        axes_h = ax.get_window_extent(renderer).height
        row_h = 1.7 * fontsize * fig.dpi / 72
        rows = max(1, int(axes_h // row_h) - (1 if title else 0))
        ncol = -(-len(handles) // rows)
        leg = ax.legend(handles, short_labels, loc='upper left', bbox_to_anchor=(1.01, 1),
                        borderaxespad=0, fontsize=fontsize, ncol=ncol, frameon=False,
                        columnspacing=1.0, handlelength=1.5,
                        title=title, title_fontsize=fontsize + 1, alignment='left')
        leg.set_in_layout(False)
        width_frac = leg.get_window_extent(renderer).width / fig_w + 0.02
        if width_frac <= max_width_frac or fontsize == 6:
            break
        leg.remove()
    fig.tight_layout(rect=(0, 0, max(1 - width_frac, 1 - max_width_frac), 1))

    if not getattr(fig, '_legend_resize_cid', None):
        def _on_resize(_event):
            place_legend_outside(fig, ax, max_width_frac)
            fig.canvas.draw_idle()
        fig._legend_resize_cid = fig.canvas.mpl_connect('resize_event', _on_resize)


class EdfFiles:
    def __init__(self, files, maskfile=None,invertmask=False):
        """
        files: array of paths to files
        maskfile: path to mask file
        invertmask: set True for pyFAI masks
        This class is designed to handle edf files, including lineEraser files (single sample, multiple files)
        Default mask corresponds to LGC convention (=0 if masked), for pyFAI masks use invertmask=True
        """
        # several files can be used as input when multiple files have been acquired for same specimen/position (and perfrom averaging of data and transmissions)
        self.files = files
        self.folder=os.path.dirname(self.files[0])
        # check if data corresponds to lineEraser case
        if 'vd' in self.files[0].split('_'):
            self.lineEraser=True
            # Redefine x and y centers from individual images
            file1,file2=self.getindividualfiles_lineEraser(self.files[0])
            im1=fabio.open(file1)
            header1=im1.header
            im2=fabio.open(file2)
            header2=im2.header
            
            self.x_center=float(header1['Center_1'])
            date = header1['Date']
            annee = int(date[:4])
            if annee<2025: # Upgrade soft
                print('self.y_center=moyenne')
                self.y_center=(float(header2['Center_2'])+float(header1['Center_2']))/2
            else:
                self.y_center=float(header2['Center_2'])
                #self.x_center=float(header2['Center_1'])
                #self.x_center=(float(header2['Center_1'])+float(header1['Center_1']))/2
                print('self.y_center=header2 check if ok by sector integration -90 et +90')
            #self.y_center=(float(header2['Center_2'])+float(header1['Center_2']))/2
            
        else:
            self.lineEraser=False
        self.nb_files = len(self.files)
        self.ismask = maskfile is not None
        self.maskfile = maskfile if self.ismask else None
        if self.ismask:
            mask=fabio.open(self.maskfile)
            self.maskdata=mask.data
        if invertmask:
            self.maskdata = self.maskdata.astype(np.int16)
            self.maskdata*=-1
            self.maskdata+=1
        print(self.files[0])
        self.image = fabio.open(self.files[0])
        header = self.image.header
        
        if self.nb_files == 1:
            self.data = self.image.data
            try:
                self.transmission = float(header['TransmittedFlux'])
            except:
                self.transmission = 1
        else:
            self.data = self.average_intensity()
            try:
                self.transmission = self.average_transmission()
            except:
                self.transmission = 1
        
        # Experimental details
        self.wl = float(header['WaveLength'])
        self.number_pixel_x, self.number_pixel_y = self.data.shape
        if not self.lineEraser:
            self.x_center = float(header['Center_1'])
            self.y_center = float(header['Center_2'])
        self.pix_size = float(header['PSize_1'])
        self.distance = float(header['SampleDistance'])
        self.D=self.distance
        self.number_pixel_x, self.number_pixel_y = self.data.shape
        try:
            self.samplename = header['Comment']
        except:
            self.samplename = ''
        try:
            file_numbers=[]
            for file in self.files:
                number=file.split('/')[-1].split('.')[0].split('_')[-1]
                if '-' in number:# line Eraser case
                    number=number.split('-')[0]
                file_numbers.append(number)
            if len(file_numbers)==1:
                self.number=file_numbers[0]
            else:
                self.number=f'{int(file_numbers[0]):03d}-{int(file_numbers[-1]):03d}'
        except:
            self.number = 0
                

   
    def writeedf(self,name,data,comment):
           
        header={'SampleDistance':self.distance,
                'WaveLength':self.wl,
                'Dim_1':data.shape[0],
                'Dim_2':data.shape[1],
                'PSize_1':self.pix_size,
                'PSize_2':self.pix_size,
                'Center_1':self.x_center,
                'Center_2':self.y_center,
                'TransmittedFlux':self.transmission,
                'Comment': comment}
        
        obj = fabio.edfimage.EdfImage(header=header,data=data)
        obj.write(name) 
    
    def average_intensity(self):
        images = [fabio.open(file).data for file in self.files]
        totaldata = np.stack(images)
        return np.mean(totaldata, axis=0)
    
    def average_transmission(self):
        trans_array = np.array([float(fabio.open(file).header['TransmittedFlux']) for file in self.files])
        return np.mean(trans_array)
    
    def getindividualfiles_lineEraser(self, file):
        directory = os.path.dirname(file)
        filename = os.path.basename(file).split('.')[0]

        # Cherche un motif du type 123456-789012
        match = re.search(r'(\d+)-(\d+)', filename)
        if not match:
            raise ValueError(f"Aucun motif numérique xxxxxx-yyyyyy trouvé dans : {filename}")

        num1, num2 = match.groups()

        # Cherche le prefix "nom_0" de manière générique :
        # On prend tout ce qui est avant le motif "_vd_"
        prefix_match = re.search(r'(.+?)_vd_', filename)
        if prefix_match:
            prefix = prefix_match.group(1)
        else:
            # Fallback : on prend le premier segment + "_0"
            prefix = filename.split('_')[0] + "_0"

        file1 = f"{directory}/{prefix}_{int(num1):05d}.edf"
        file2 = f"{directory}/{prefix}_{int(num2):05d}.edf"

        return file1, file2
    
    def ai_from_edf(self):        
        detector = Detector(pixel1=self.pix_size, pixel2=self.pix_size)
        ai = AzimuthalIntegrator(dist=self.distance, detector=detector) 
        ai.setFit2D(self.distance * 1000, self.x_center, self.y_center, wavelength=self.wl * 1e10)
        return ai

    def integrate(self, remove=False):
        path = os.path.dirname(self.files[0])
        output = os.path.join(path, self.samplename + "_radial_profile.dat")
        nbins = 1000
        unit_type = "q_A^-1"
        
        self.ai = self.ai_from_edf()
        #mask = fabio.open(self.maskfile).data if self.maskfile else None
        
        q, i = self.ai.integrate1d(self.data, nbins, filename=output, mask=self.maskdata, unit=unit_type, normalization_factor=1)
        if remove:
            os.remove(output)
        return q, i
    
    def save_plot(self):
        # integrate and save
        q,i=self.integrate(self)
        # plot
        plt.figure()
        plt.loglog(q,i)
        plt.xlabel('Q ($\\AA^{-1}$)')
        plt.ylabel('Intensity')
        plt.title(self.samplename)
        plt.show()

    def calc_azimprofile(self,qvalue,threshold):
        """
        Method to extract azimuthal profile at a given q value
        qvalue: float, q value at which the azimuthal profile should be extracted
        threshold: float, tolerance around q value
        """
        path = os.path.dirname(self.files[0])
        output = os.path.join(path, self.samplename + f"_azim_profile_q={qvalue:.3f}.dat")
        nbins = 1000
        unit_type = "q_A^-1"
        
        self.ai = self.ai_from_edf()
        #mask = fabio.open(self.maskfile).data if self.maskfile else None
        
        chi,I=self.ai.integrate_radial(self.data, 540,mask=self.maskdata, radial_range=(qvalue*(1-threshold), qvalue*(1+threshold)), radial_unit=unit_type,method=("no", "histogram", "cython"))
        
        return chi, I

    ## Functions for nematic order parameter
    def pseudo_voigt(self,x,y0,I, x0, gamma,eta,slope):
        pi=np.pi
        ln2=np.log(2)
        a=(2/gamma)*(ln2/pi)**(1/2)
        b=(4*ln2/(gamma**2))
        return y0+slope*x+I*(eta*(a*np.exp(-b*(x-x0)**2))+(1-eta)*((1/pi)*((gamma/2)/((x-x0)**2+(gamma/2)**2))))
    
    def myfunc(self,x,I, x0, gamma,eta):
        # f(beta)sin(beta)
        return self.pv_nobckgd(x, I, x0, gamma,eta)*np.sin(x)
    
    def P2_nobckgd(self,x,I, x0, gamma,eta):
        return ((1/2)*(3*np.cos(x)*np.cos(x)-1)*self.pv_nobckgd(x, I, x0, gamma,eta)*np.sin(x))
    
    def pv_nobckgd(self,x,I, x0, gamma,eta):
        pi=np.pi
        ln2=np.log(2)
        a=(2/gamma)*(ln2/pi)**(1/2)
        b=(4*ln2/(gamma**2))
        return I*(eta*(a*np.exp(-b*(x-x0)**2))+(1-eta)*((1/pi)*((gamma/2)/((x-x0)**2+(gamma/2)**2))))
     
    def calc_S(self,x,I,x0,gamma,eta):
        # calculate degree of order
        from scipy.integrate import quad
        norm_factor , error = quad(self.myfunc,0,np.pi,args=(I,x0*np.pi/180,gamma*np.pi/180,eta))
        S , error = quad(self.P2_nobckgd,0,np.pi,args=(I,x0*np.pi/180,gamma*np.pi/180,eta))
        S/=norm_factor
        return S
    
    def compute_S(self,qvalue,threshold):
        """
        performs fit of azimuthal profile for each reflection of a single image
        results are stored in a dictionnary {reflection:[y0,I,mean,sigma,aire]}
        """
        from scipy.signal import savgol_filter
        from scipy.optimize import curve_fit
        profile=self.pyFAI_extract_azimprofiles(qvalue=qvalue,threshold=threshold)
        
        # Smooth data using Savitzky-Golay filter

        x=profile[0];y=savgol_filter(profile[1],5,1)
        # find position of max and define fitting range with "width" variable
        index=np.argmax(y[x<150]) #index of max Intensity
        width=90
        a=x[index]-width/2; b=x[index]+width/2 
        
        # define fitting region in x
        test=np.argwhere((a<x)&(x<b))
        amin=test[0,0] #index where x is close to a
        amax=test[-1,0] # index where x is close to b
        #extract arrays corresponding to the fitting region
        x2fit=x[amin:amax+1]
        x2fit = np.asarray(x2fit)
        xmin=np.min(x2fit);xmax=np.max(x2fit)
        y2fit=y[test[:,0]] 
        y2fit=np.asarray(y2fit)
        # give a first estimation of the fitted parameters
        y0_guess=np.mean(y[amin:amin+5]) # flat horizonthal background
        x0_guess=x[index]
        I_guess=np.max(y)
        gamma_guess=5
        eta_guess=1
        slope_guess=0.001
        init_params=[y0_guess,I_guess,x0_guess,gamma_guess,eta_guess,slope_guess]
        
        # define bounds
        lb_G=[0,0,xmin,0,0,-np.inf] # sigma low bounds can be negative in the formula 
        ub_G=[np.inf,np.inf,xmax,np.inf,1,np.inf]
        bounds_G=(lb_G,ub_G) 
        
        # fit the parameters and extract sigmas

        try:
            params, _ =curve_fit(self.pseudo_voigt,x2fit,y2fit,p0=init_params,bounds=bounds_G,method='trf')
            y0,I,x0,gamma,eta,slope=params
            
            if x0<0 and x0 <-20:
                x0_S = x0 + 180 # position must be between 0 and 180 (we only consider this range for S calculation)
            else:
                x0_S = x0
            if x0_S >20:
                x0_S -=90 # center position around 0 so that 0<S<1 
            
            # Calculate R² (coefficient of determination)
            residuals = y2fit - self.pseudo_voigt(x2fit, *params)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y2fit - np.mean(y2fit))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # calculate degree of order
            S = self.calc_S(x,I,x0_S,gamma,eta)
        except Exception as e:
            print(f"Fit failed for q = {qvalue}: {e}")
            y0=np.nan
            I=np.nan
            mean=np.nan
            sigma=np.nan
            aire=np.nan
            S=np.nan
            r_squared=np.nan
            fitflag=False
                           
        return S,r_squared

    def compute_offset(self,
                       q0=0.05, dq=0.005, npt_rad=1440, npt_azim=1440,
                       plot_profile=False, plot_image=False):
        """
        Trouve l'angle azimutal maximal à q fixé et trace la droite sur l'image SAXS.

        Parameters
        ----------
        q0 : float
            Valeur de q (Å⁻¹) autour de laquelle extraire le profil.
        dq : float
            Largeur de la fenêtre radiale (Å⁻¹).
        npt_rad : int
            Nombre de bins radiaux pour l'intégration 2D.
        npt_azim : int
            Nombre de bins azimutaux.
        plot_profile : bool
            Si True, trace le profil I(χ) et marque le maximum.
        plot_image : bool
            Si True, trace la droite correspondant à l'angle maximal sur l'image.

        Returns
        -------
        max_angle : float
            Angle azimutal (°) où l’intensité est maximale à q0. Valeur comprise entre -45° et +45°
            
        
        """
        # Charger l'image
        
        ny, nx = self.data.shape
        ai = self.ai_from_edf()
        # Intégration 2D
        Iqchi, q, chi = ai.integrate2d(
            self.data,
            npt_azim,
            npt_rad,
            unit=("q_A^-1", "chi_deg"),
            method="bbox"
        )

        # Sélection de la plage q0 ± dq
        mask_q = (q >= q0 - dq) & (q <= q0 + dq)
        if not np.any(mask_q):
            raise ValueError(f"Aucune valeur de q trouvée dans [{q0-dq}, {q0+dq}]")
        #print(len(mask_q),Iqchi.shape[0])
        Iq = np.mean(Iqchi[mask_q, :], axis=0)
        Ichi = np.mean(Iqchi, axis =1)

        # Trouver le maximum
        max_idx = np.argmax(Ichi)
        max_angle = chi[max_idx] %90
        if max_angle >45:
            max_angle -=90
        return max_angle
        
        

    

class CalibrationSet:
    def __init__(self, samplefiles, I_0, referencefiles=None, maskfile=None,invertmask=False):
        """
        samplefiles: array of paths to files for sample measurement
        referencefiles: array of paths to files for sample reference measurement
        maskfile: path to mask file (edf format)
        invertmask: set True for pyFAI masks
        
        """
        self.I_0 = I_0
        self.ismask = maskfile is not None
        self.maskfile = maskfile if self.ismask else None
        if self.ismask:
            mask=fabio.open(self.maskfile)
            self.maskdata=mask.data
        if invertmask:
            self.maskdata*=-1
            self.maskdata+=1
        self.samplefiles = EdfFiles(samplefiles, maskfile)
        

        # Correct sample data for transmission
        self.sampledata = self.samplefiles.data / (self.samplefiles.transmission / I_0)
        
        # Experimental details
        self.wl = self.samplefiles.wl
        self.x_center = self.samplefiles.x_center
        self.y_center = self.samplefiles.y_center
        self.pix_size = self.samplefiles.pix_size
        self.distance = self.samplefiles.distance
        self.number_pixel_x = self.samplefiles.number_pixel_x
        self.number_pixel_y = self.samplefiles.number_pixel_y
        
        if referencefiles:
            ref = EdfFiles(referencefiles,self.maskfile)
            # transmission correction
            self.refdata = ref.data / (ref.transmission / I_0)
            
            # perform reference subtraction
            self.data = self.sampledata - self.refdata
        else:
            self.data=self.sampledata

        
    
    def ai_from_edf(self):        
        return self.samplefiles.ai_from_edf()
    
    def integrate(self, remove=False):
        path = os.path.dirname(self.samplefiles.files[0])
        output = os.path.join(path, self.samplefiles.samplename + "_radial_profile.dat")
        nbins = 1000
        unit_type = "q_A^-1"
        
        self.ai = self.ai_from_edf()
               
        q, i = self.ai.integrate1d(self.data, nbins, filename=output, mask=self.maskdata, unit=unit_type, normalization_factor=1)
        if remove:
            os.remove(output)
        return q, i

class SAXSDataSet():
    def __init__(self, samplefiles, I_0, referencefiles=None,calibsamplefiles=None,calibreferencefiles=None,calibrant=None, I_calibrant=1, maskfile=None,invertmask=False,k=1,refineflag=False, qseuil=0.8, plotrefine=False, force_distance=None):
        """
        Class to describe a SAXSDataSet (single sample, several files), inclunding intensity calibration, transmission correction, and radial profile calculation.
        samplefiles: array of paths to files for sample measurement
        I_0: intensity of the incident beam
        referencefiles: array of paths to files for sample reference measurement
        calibsamplefiles: array of paths to files for calibrant measurement
        calibreferencefiles: array of paths to files for calibrant reference measurement
        calibrant: str Nature of calibrant (water, hexane)
        I_calibrant: float theoretical value of I for calibrant
        maskfile: path to mask file (edf format)
        invertmask: set True for pyFAI masks
        k: coefficient by which reference data should be multiplied
        refineflag: bool (False) used to estiamte k coeff. for reference substraction
        qseuil: float q value beyond which background substration is refined (used when refineflag=True)
        plotrefine: bool (False) used to plot refined net data
        force_distance: float or None — if not None, overrides the sample-to-detector distance (in m) read from the EDF header
        """
        self.I_0 = I_0
        # set mask
        self.ismask = maskfile is not None
        self.maskfile = maskfile if self.ismask else None
        if self.ismask:
            mask=fabio.open(self.maskfile)
            self.maskdata=mask.data
            self.maskdata=1-self.maskdata
            if invertmask:
                self.maskdata=1-self.maskdata

        self.samplefiles = EdfFiles(samplefiles, maskfile)
        # correct sample data with transmission
        self.sampledata = self.samplefiles.data / (self.samplefiles.transmission / I_0)
        
        ## Perform Calibration using Calibration dataset
        self.calibration_coef=1
        # theoretical value
        if calibrant=='water' or calibrant=='H2O':
            I_calibrant=0.0162
            
        elif calibrant=='hexane':
            I_calibrant=0.028
        else:
            I_calibrant=I_calibrant
        # Compute calibration coef = I_calibrant/(mean(plateau))
        if calibsamplefiles is not None and calibreferencefiles is not None:
            self.calibset=CalibrationSet(calibsamplefiles,I_0,calibreferencefiles,maskfile=self.maskfile)
            self.qcalib,self.icalib=self.calibset.integrate()
            # select data corresponding to q<0.5 (plateau)
            Icalibok=[self.icalib[j] for j in range(len(self.qcalib)) if self.qcalib[j]<0.5]
            icalibmean=np.mean(Icalibok)
            self.calibration_coef=I_calibrant/icalibmean 
            self.absolute=True
        else:
            self.calibration_coef=1  
            self.absolute=False   
               
        ## deal with referencefiles
        self.k = k
        self.refine_info = None
        if referencefiles:
            self.referencefiles = EdfFiles(referencefiles,maskfile)
            
            # correct reference data with transmission
            self.refdata = self.referencefiles.data / (self.referencefiles.transmission / I_0)
            
            if refineflag:
                self.qseuil = qseuil
                self.plotrefine = plotrefine
                self.k, qmax_ref, qsample_ref = self.refine()
                self.refine_info = (self.k, qmax_ref, qsample_ref)
                self.refdata *= self.k
            else:
                self.refdata *= self.k
        else:
            self.refdata = np.zeros_like(self.sampledata)
        
        # compute calibrated corrected intensity
        self.data = self.calibration_coef*(self.sampledata - self.refdata)
        
        # Experimental details
        self.wl = self.samplefiles.wl
        self.x_center = self.samplefiles.x_center
        self.y_center = self.samplefiles.y_center
        self.pix_size = self.samplefiles.pix_size
        self.distance = self.samplefiles.distance
        self.D=self.distance
        self.number_pixel_x = self.samplefiles.number_pixel_x
        self.number_pixel_y = self.samplefiles.number_pixel_y
        self.folder = self.samplefiles.folder
        self.samplename=self.samplefiles.samplename
        self.number=self.samplefiles.number

        # override distance if requested by the user
        if force_distance is not None:
            self.distance = float(force_distance)
            self.D = self.distance
            self.samplefiles.distance = self.distance
            self.samplefiles.D = self.distance

        # perform caving
        if self.ismask:
            self.caving()

    def qmap(self, qmin=0, qmax=0.2, rotate=False, resolution=1000):
        """Normalized intensity regridded on a regular (Qx, Qz) grid for qmin <= |q| <= qmax: (QX, QZ, Z)."""
        di = ((np.arange(self.number_pixel_y) - self.x_center) * self.pix_size)[None, :]
        dj = ((np.arange(self.number_pixel_x) - self.y_center) * self.pix_size)[:, None]
        denom = np.sqrt(self.D**2 + di**2 + dj**2)
        a = 2 * np.pi / self.wl
        qx = (a * di / denom) * 1e-10
        qy = (a / denom) * (self.D - denom) * 1e-10
        qz = (a * dj / denom) * 1e-10

        qnorm = np.sqrt(qx**2 + qy**2 + qz**2)
        mask = (qnorm >= qmin) & (qnorm <= qmax)
        qx_masked, qz_masked, intensity = qx[mask], qz[mask], self.data[mask]
        if len(qx_masked) < 4:
            raise ValueError(f"Too few detector pixels between qmin={qmin} and qmax={qmax} to build a 2D image.")
        intensity = intensity / np.nanmax(intensity)

        qx_lin = np.linspace(qx_masked.min(), qx_masked.max(), resolution)
        qz_lin = np.linspace(qz_masked.min(), qz_masked.max(), resolution)
        QX, QZ = np.meshgrid(qx_lin, qz_lin)
        Z = griddata((qx_masked, qz_masked), intensity, (QX, QZ), method='linear')
        if rotate:
            Z = np.rot90(Z, k=-1)  # -1 pour sens horaire
            QZ, QX = np.rot90(QX, k=-1), np.rot90(QZ, k=-1)
        return QX, QZ, Z

    @staticmethod
    def draw_qmap(fig, ax, QX, QZ, Z, qmax=0.2, vmin=-3, vmax=0, cmap='turbo', qcircles=None,
                  draw_sectors=False, nb_azimuths=2, width=10, offset=0, delta=90, fontsize=12):
        """Draw a qmap() result on ax, with optional constant-q circles and integration sectors (half width `width`)."""
        norm = LogNorm(vmin=10**(vmin), vmax=10**vmax)
        mesh = ax.pcolormesh(QX, -QZ, Z, shading='auto', cmap=cmap, norm=norm)
        ax.set_xlabel(r"$Q_x (\AA^{-1})$", fontsize=fontsize)
        ax.set_ylabel(r"$Q_z (\AA^{-1})$", fontsize=fontsize)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label("Normalized Intensity", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        ax.set_aspect('equal')
        fig.tight_layout()

        def format_coord(x, z):
            q = math.hypot(x, z)
            # The image is drawn against -Qz and the sector wedges at -azimuth, so
            # negating the on-screen angle gives the same chi as the azimuthal profile.
            parts = [f"q={q:.4f} A-1"]
            if q > 0:
                parts.append(f"d={2 * np.pi / q / 10:.3f} nm")
            parts.append(f"chi={-math.degrees(math.atan2(z, x)):.1f}deg")
            return "   ".join(parts)

        ax.format_coord = format_coord

        if qcircles is not None:
            colors = ['black','purple','pink','palegreen']
            # prendre les limites effectives de l'axe (après pcolormesh)
            xmin, xmax = sorted(ax.get_xlim())
            zmin, zmax = sorted(ax.get_ylim())
            for i, q_val in enumerate(qcircles):
                q_val = float(q_val)
                theta = np.linspace(0, 2*np.pi, 2000)
                x_circle = q_val * np.cos(theta)
                # adapter l'ordonnée au système de l'affichage (on a tracé -QZ)
                z_plot = -q_val * np.sin(theta)
                # garder uniquement les points du cercle qui tombent dans la fenêtre affichée
                mask = (x_circle >= xmin) & (x_circle <= xmax) & (z_plot >= zmin) & (z_plot <= zmax)
                if not np.any(mask):
                    continue
                ax.plot(x_circle[mask], z_plot[mask], linestyle='dashed', color=colors[i % len(colors)], linewidth=2)
                x_text = -q_val
                z_text = q_val*(-1)**i
                if x_text<xmin or x_text>xmax:
                    x_text = -x_text
                if z_text<zmin or z_text>zmax:
                    z_text=-z_text
                ax.text(x_text, z_text + 0.002, f"{q_val:.3f}",
                        color=colors[i % len(colors)], fontsize=fontsize - 2,
                        ha='center', va='bottom',
                        bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=1),
                        clip_on=True)

        if draw_sectors:
            wedge_colors = ['red', 'lime', 'cyan', 'yellow', 'magenta', 'orange', 'blue']
            for n in range(nb_azimuths):
                az = ((offset + n * delta + 180) % 360) - 180
                min_az = az - width
                max_az = az + width
                color = wedge_colors[n % len(wedge_colors)]
                # Wedge: angles in degrees CCW from +x, inverted for -Qz
                ax.add_patch(Wedge((0.0, 0.0), qmax, -max_az % 360, -min_az % 360,
                                   facecolor='none', edgecolor=color, linestyle='-',
                                   linewidth=1.2, alpha=0.9, zorder=5))
                for th in (min_az, max_az):
                    th_rad = math.radians(-th)
                    ax.plot([0.0, qmax * math.cos(th_rad)], [0.0, qmax * math.sin(th_rad)],
                            linestyle='-', color=color, linewidth=1.2, alpha=0.9, zorder=6)
                bisector_rad = math.radians(-az)
                ax.text(0.75 * qmax * math.cos(bisector_rad), 0.75 * qmax * math.sin(bisector_rad),
                        f"{az:.0f}°\n±{width:.0f}°", color=color,
                        fontsize=fontsize - 3, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), zorder=7)
            ax.set_aspect('equal', adjustable='box')

    def plot2d_vsq(
            self,
            qmin:float = 0,
            qmax:float = 0.2,
            vmin:float = -3,
            vmax:float = 0,
            qcircles:tuple=None,
            show:bool=False,
            rotate:bool=False,
            cmap:str = 'turbo',
            draw_sectors:bool=False,
            nb_azimuths: int = 2,
            width: float = 10,      # half-width in degrees (same convention as sector_integration)
            offset: float = 0,      # degrees
            delta: float = 90,      # degrees between sector centers
            save_npz: bool = False,
            qmap_result=None
        ):
        QX, QZ, Z = qmap_result if qmap_result is not None else self.qmap(qmin, qmax, rotate)
        fig = plt.figure(dpi=200)
        self.draw_qmap(fig, fig.gca(), QX, QZ, Z, qmax=qmax, vmin=vmin, vmax=vmax, cmap=cmap, qcircles=qcircles,
                       draw_sectors=draw_sectors, nb_azimuths=nb_azimuths, width=width, offset=offset, delta=delta)

        outputdir = self.folder +'/png_images/'
        os.makedirs(outputdir,exist_ok=True)
        if save_npz:
            npyfile=outputdir+f'{self.samplename}_Img_{self.number}'
            np.savez(npyfile,Qx=QX, QZ=-QZ, Z=Z)

        figname = outputdir + f'{self.samplename}_Img_{self.number}.png'
        fig.savefig(figname)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        return figname

    def caving(self,max_iter=10):
        """
        Remplace les pixels masqués par la valeur symétrique
        (par rapport à x_center, z_center) en plusieurs passes.

        max_iter : nombre maximal d'itérations
        """
        self.data = np.where(self.maskdata == 1.0, np.nan, self.data)

        for it in range(max_iter):
            modified = False
            for x in range(int(self.number_pixel_x)):      # lignes
                for z in range(int(self.number_pixel_y)):  # colonnes
                    if np.isnan(self.data[x, z]):       # pixel masqué
                        xsym = int(2 * self.y_center - x)
                        zsym = int(2 * self.x_center - z)

                        # Vérifier les bornes
                        if 0 <= xsym < int(self.number_pixel_x) and 0 <= zsym < int(self.number_pixel_y):
                            if not np.isnan(self.data[xsym, zsym]):  # pixel symétrique valide
                                self.data[x, z] = self.data[xsym, zsym]
                                self.maskdata[x,z] = 0 # update mask
                                modified = True
            if not modified:
                break
        
    
    
    def refine(self):
        # determine coefficient and return q values at which the solvent peak is observed in both ref and sample measurement
        qref, iref = self.referencefiles.integrate(remove=True)
        iref*=self.calibration_coef
        sample = EdfFiles(self.samplefiles.files, self.maskfile)
        qsample, isample = sample.integrate(remove=True)
        isample*=self.calibration_coef
        index_ref = np.searchsorted(qref, self.qseuil)
        index_sample = np.searchsorted(qsample, self.qseuil)
        
        if index_ref >= len(iref) or index_sample >= len(isample):
            return 1, np.nan, np.nan  # Avoid division errors if out of range
        
        iref_max = np.max(iref[index_ref:])
        index_ref=np.argmax(iref[index_ref:])
        qmax_ref=qref[index_ref]

        isample_max = np.max(isample[index_sample:])  
        index_sample=np.argmax(isample[index_sample:])
        qmax_sample=qref[index_sample]              
        
        self.k = 0.95*(isample_max / self.samplefiles.transmission) / (iref_max / self.referencefiles.transmission)
        if np.abs(qmax_ref-qmax_sample) > 0.05 * qmax_ref: # solvent peaks do not overlap by 5%, decrease self.k
            self.k *= 0.9
        if self.plotrefine:
            icorr = self.calibration_coef*((isample / (self.samplefiles.transmission/self.I_0)) - self.k * (iref / (self.referencefiles.transmission/self.I_0)))
            plt.figure()
            plt.loglog(qref, (iref / (self.referencefiles.transmission/self.I_0)), label='Ref-Transmission corrected')
            plt.loglog(qref, self.k*(iref / (self.referencefiles.transmission/self.I_0)), label='k * Ref-Transmission corrected')
            plt.loglog(qsample, isample / (self.samplefiles.transmission/self.I_0), label='Sample-Transmission corrected')
            plt.loglog(qsample, icorr, label=f'Corrected intensity, k={self.k:.3f}')
            plt.xlabel('Q $(\\AA^{-1})$')
            if self.absolute:
                plt.ylabel('Intensity $(cm^{-1}) $')
                
            else:
                plt.ylabel('Intensity')
            
            plt.legend()
            plt.show()
        
        return  self.k, qmax_ref,qmax_sample

    def ai_from_edf(self):        
        return self.samplefiles.ai_from_edf()
    
    def save_corrected_image(self):
        
        header = {
            'WaveLength': self.wl,
            'SampleDistance' : self.D,
            'Center_1': self.x_center,
            'Center_2':self.y_center,
            'PSize_1': self.pix_size,
            'SampleDistance':self.D,
            'Comment':self.samplename
        }   
        outputpath = f'{self.folder}/corrected_edf/'
        os.makedirs(outputpath, exist_ok = True)
        filename = outputpath+f'{self.samplename}_Img_{self.number}.edf'
        # Create an EDF image object
        edf_img = fabio.edfimage.edfimage(data=self.data, header=header)

        # Write to file
        edf_img.write(filename)

    def print_values(self):
        line2write='--------------- Sample details -----------------\n'
        line2write+=f'Sample name: {self.samplefiles.samplename}\n'
        line2write+=f'Intensity calibration coefficient={self.calibration_coef:.6f}\n\n'
        line2write+='------------ Experiment details ----------------\n'
        line2write += f'λ = {1e10*self.wl:.5f} Å\n'
        line2write+=f'Incident intensity={self.I_0} cps\n'
        line2write+=f'distance={self.distance}m\n'
        line2write+=f'beam_center=({self.samplefiles.x_center},{self.samplefiles.y_center})\n\n'
        line2write+='------------- Detector details -----------------\n'
        line2write+=f'pixel_size={self.samplefiles.pix_size}m\n'
        line2write+=f'image_size={self.samplefiles.number_pixel_x} X {self.samplefiles.number_pixel_y}pixels\n'
        print(line2write)

    @property
    def label(self):
        return f"{self.samplename}_Img{self.number}"

    def radial_profile(self, azimuth=0, width=360, nbins=2000):
        """I(q) in the sector centred on azimuth (°) of full width (°), clamped to [-180, 180] as in SAXS explorer; width >= 360 is the whole ring."""
        azimuth_range = None
        if width < 360:
            az = ((azimuth + 180) % 360) - 180
            azimuth_range = (max(az - width / 2, -180), min(az + width / 2, 180))
        q, i = self.ai_from_edf().integrate1d(self.data, nbins, azimuth_range=azimuth_range, mask=self.maskdata,
                                              unit="q_A^-1", normalization_factor=1)
        return q, i

    def sector_profiles(self, nb_azimuths=2, width=10, offset=0, delta=90, nbins=2000):
        """[(azimuth, q, I)] for nb_azimuths sectors of half width `width` (°), centred on offset + n*delta."""
        ai = self.ai_from_edf()
        profiles = []
        for n in range(nb_azimuths):
            azimuth = (offset + n * delta) % 360
            if azimuth > 180:
                azimuth -= 360
            azimuth_range = (max(azimuth - width, -180), min(azimuth + width, 180))
            q, i = ai.integrate1d(self.data, nbins, azimuth_range=azimuth_range, mask=self.maskdata,
                                  unit="q_A^-1", normalization_factor=1)
            profiles.append((azimuth, q, i))
        return profiles

    def azimuthal_profile(self, qvalue, threshold, nbins=540):
        """I(chi) integrated over the ring q in [qvalue - threshold, qvalue + threshold] (Å⁻¹), as in SAXS explorer."""
        chi, i = self.ai_from_edf().integrate_radial(self.data, nbins, mask=self.maskdata,
                                                     radial_range=(qvalue - threshold, qvalue + threshold),
                                                     radial_unit="q_A^-1", unit="chi_deg")
        return chi, i

    def save_profile(self, subdir, filename, x, y, columns, description):
        path = os.path.join(self.folder, subdir)
        os.makedirs(path, exist_ok=True)
        output = os.path.join(path, filename)
        header = (f"{description}\nSample: {self.samplename}\nFiles: {', '.join(os.path.basename(f) for f in self.samplefiles.files)}\n"
                  f"I_0={self.I_0} k={self.k} calibration_coef={self.calibration_coef}\n{columns}")
        np.savetxt(output, np.column_stack([x, y]), header=header)
        return output

    def calc_azimprofile(self,qvalue,threshold):
        """
        Method to extract azimuthal profile at a given q value
        qvalue: float, q value at which the azimuthal profile should be extracted
        threshold: float, tolerance around q value
        """
        path = os.path.dirname(self.files[0])
        output = os.path.join(path, self.samplename + f"_azim_profile_q={qvalue:.3f}.dat")
        nbins = 1000
        unit_type = "q_A^-1"
        self.ai = self.ai_from_edf()
        
        chi,I=self.ai.integrate_radial(self.data, 540,mask=self.maskdata, radial_range=(qvalue*(1-threshold), qvalue*(1+threshold)), radial_unit=unit_type,method=("no", "histogram", "cython"))
        np.savetxt(output,np.column_stack[chi,I])
        return chi, I

    ## Functions for nematic order parameter
    def pseudo_voigt(self,x,y0,I, x0, gamma,eta,slope):
        pi=np.pi
        ln2=np.log(2)
        a=(2/gamma)*(ln2/pi)**(1/2)
        b=(4*ln2/(gamma**2))
        return y0+slope*x+I*(eta*(a*np.exp(-b*(x-x0)**2))+(1-eta)*((1/pi)*((gamma/2)/((x-x0)**2+(gamma/2)**2))))
    
    def myfunc(self,x,I, x0, gamma,eta):
        # f(beta)sin(beta)
        return self.pv_nobckgd(x, I, x0, gamma,eta)*np.sin(x)
    
    def P2_nobckgd(self,x,I, x0, gamma,eta):
        return ((1/2)*(3*np.cos(x)*np.cos(x)-1)*self.pv_nobckgd(x, I, x0, gamma,eta)*np.sin(x))
    
    def pv_nobckgd(self,x,I, x0, gamma,eta):
        pi=np.pi
        ln2=np.log(2)
        a=(2/gamma)*(ln2/pi)**(1/2)
        b=(4*ln2/(gamma**2))
        return I*(eta*(a*np.exp(-b*(x-x0)**2))+(1-eta)*((1/pi)*((gamma/2)/((x-x0)**2+(gamma/2)**2))))
     
    def calc_S(self,x,I,x0,gamma,eta):
        # calculate degree of order
        from scipy.integrate import quad
        norm_factor , error = quad(self.myfunc,0,np.pi,args=(I,x0*np.pi/180,gamma*np.pi/180,eta))
        S , error = quad(self.P2_nobckgd,0,np.pi,args=(I,x0*np.pi/180,gamma*np.pi/180,eta))
        S/=norm_factor
        return S
    
    def compute_S(self,qvalue,threshold):
        """
        performs fit of azimuthal profile for each reflection of a single image
        returns S, r_squared
        """
        from scipy.signal import savgol_filter
        from scipy.optimize import curve_fit
        profile=self.pyFAI_extract_azimprofiles(qvalue=qvalue,threshold=threshold)
        
        # Smooth data using Savitzky-Golay filter
        x=profile[0];y=savgol_filter(profile[1],5,1)
        # find position of max and define fitting range with "width" variable
        index=np.argmax(y[x<150]) #index of max Intensity
        width=90
        a=x[index]-width/2; b=x[index]+width/2 
        
        # define fitting region in x
        test=np.argwhere((a<x)&(x<b))
        amin=test[0,0] #index where x is close to a
        amax=test[-1,0] # index where x is close to b
        #extract arrays corresponding to the fitting region
        x2fit=x[amin:amax+1]
        x2fit = np.asarray(x2fit)
        xmin=np.min(x2fit);xmax=np.max(x2fit)
        y2fit=y[test[:,0]] 
        y2fit=np.asarray(y2fit)
        # give a first estimation of the fitted parameters
        y0_guess=np.mean(y[amin:amin+5]) # flat horizonthal background
        x0_guess=x[index]
        I_guess=np.max(y)
        gamma_guess=5
        eta_guess=1
        slope_guess=0.001
        init_params=[y0_guess,I_guess,x0_guess,gamma_guess,eta_guess,slope_guess]
        
        # define bounds
        lb_G=[0,0,xmin,0,0,-np.inf] # sigma low bounds can be negative in the formula 
        ub_G=[np.inf,np.inf,xmax,np.inf,1,np.inf]
        bounds_G=(lb_G,ub_G) 
        
        # fit the parameters and extract sigmas

        try:
            params, _ =curve_fit(self.pseudo_voigt,x2fit,y2fit,p0=init_params,bounds=bounds_G,method='trf')
            y0,I,x0,gamma,eta,slope=params
            
            if x0<0 and x0 <-20:
                x0_S = x0 + 180 # position must be between 0 and 180 (we only consider this range for S calculation)
            else:
                x0_S = x0
            if x0_S >20:
                x0_S -=90 # center position around 0 so that 0<S<1 
            
            # Calculate R² (coefficient of determination)
            residuals = y2fit - self.pseudo_voigt(x2fit, *params)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y2fit - np.mean(y2fit))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # calculate degree of order
            S = self.calc_S(x,I,x0_S,gamma,eta)
        except Exception as e:
            print(f"Fit failed for q = {qvalue}: {e}")
            y0=np.nan
            I=np.nan
            mean=np.nan
            sigma=np.nan
            aire=np.nan
            S=np.nan
            r_squared=np.nan
            fitflag=False
                           
        return S,r_squared




