cimport cython
from cv2 import drawContours,inRange,dilate,erode
import numpy as np
cimport numpy as np
from cpython cimport bool
ctypedef np.float_t FTYPE_t
ctypedef np.int_t DTYPE_t
DTYPE = np.int
ctypedef np.uint8_t U8TYPE_t
U8TYPE = np.uint8
cdef inline float fmax(float a, float b) nogil: return a if a >= b else b
cdef inline float fmin(float a, float b) nogil: return a if a <= b else b
cdef inline float inrange(float a, float _min, float _max) nogil: 
    return 1 if a<=_max and a>=_min else 0
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def label(np.ndarray[FTYPE_t, ndim=2] img, list contours, bool med_filt=False,
          DTYPE_t dil_size=25, DTYPE_t er_size=3,
          FTYPE_t tol=0):
    cdef int x_shape=img.shape[0]
    cdef int y_shape=img.shape[1]
    cdef np.ndarray[U8TYPE_t, ndim=2] labeled = np.zeros((x_shape,y_shape),
                                                        dtype=U8TYPE) 
    cdef int N = len(contours)
    cdef np.ndarray[U8TYPE_t, ndim=2] diltmp = np.empty((x_shape,y_shape),
                                                        dtype=U8TYPE)
    cdef np.ndarray[U8TYPE_t, ndim=2] ertmp = np.empty((x_shape,y_shape),
                                                        dtype=U8TYPE)
    cdef np.ndarray[U8TYPE_t, ndim=2] dil_mask = np.ones((dil_size,dil_size),
                                                         dtype=U8TYPE)
    cdef np.ndarray[U8TYPE_t, ndim=2] er_mask = np.ones((er_size,er_size),
                                                         dtype=U8TYPE)
    cdef np.ndarray[FTYPE_t, ndim=1] _min = np.zeros(N)
    _min.fill(10000.0)
    cdef np.ndarray[FTYPE_t, ndim=1] _max = np.zeros(N) 
    _max.fill(0.0)
    cdef np.ndarray[U8TYPE_t, ndim=2] filtered = np.zeros((x_shape,y_shape),
                                                          dtype=U8TYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] xpos = np.zeros((N,img.size),DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] ypos = np.zeros((N,img.size),DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] count = np.zeros(N,DTYPE)
    cdef int i,j,k,v
    for i in range(1,N+1):
        drawContours(labeled, contours, i-1, i, -1)
    if med_filt:
        diltmp[:,:] = dilate(labeled.copy(), dil_mask)
        ertmp[:,:] = erode(labeled.copy(), er_mask)
        with nogil:
                for j in range(x_shape):
                    for k in range(y_shape):
                        if diltmp[j,k]!=0 and (labeled[j,k]!=0 or img[j,k]!=0):
                            v=diltmp[j,k]-1
                            xpos[v,count[v]]=j
                            ypos[v,count[v]]=k
                            count[v]+=1
                            v=ertmp[j,k]-1
                            if img[j,k]!=0 and v!=-1:
                                _min[v]=fmin(img[j,k],_min[v])
                                _max[v]=fmax(img[j,k],_max[v])
                for i in range(N):
                    for j in range(count[i]-1):
                        if ((labeled[xpos[i,j],ypos[i,j]]!=0 and
                            img[xpos[i,j],ypos[i,j]]!=0)
                        or inrange(img[xpos[i,j],ypos[i,j]],(1-tol)*_min[i],
                                    (1+tol)*_max[i])):
                            filtered[xpos[i,j],ypos[i,j]]=255
        return labeled, filtered, _min, _max

    return labeled, filtered, _min, _max
