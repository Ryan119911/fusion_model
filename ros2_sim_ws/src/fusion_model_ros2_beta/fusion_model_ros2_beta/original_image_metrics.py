"""Standard 7x7 SSIM, data_range=1, sample covariance; SciPy-only ROS host."""
import numpy as np
from scipy.ndimage import uniform_filter


def ssim(original,predicted):
    x,y=np.asarray(original,np.float64),np.asarray(predicted,np.float64)
    if x.shape!=y.shape or x.ndim!=2 or min(x.shape)<7:
        raise ValueError('SSIM requires matching images at least 7x7')
    mx,my=uniform_filter(x,size=7),uniform_filter(y,size=7)
    correction=49/48
    vx=correction*(uniform_filter(x*x,size=7)-mx*mx)
    vy=correction*(uniform_filter(y*y,size=7)-my*my)
    covariance=correction*(uniform_filter(x*y,size=7)-mx*my)
    s=((2*mx*my+.01**2)*(2*covariance+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return float(s[3:-3,3:-3].mean())
