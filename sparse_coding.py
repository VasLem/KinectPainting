import numpy as np
from numpy import array, argmax, argmin, concatenate,diag
from numpy import dot, sign, zeros, zeros_like, random,trace, mean
from numpy import allclose
from numpy.linalg import inv, pinv, norm
from numpy import sum as npsum
from numpy import abs as npabs
import scipy.optimize as optimize
from scipy.optimize import fmin_tnc

class FeatureSignSearch(object):

    def __init__(self):
        self.bmat = None
        self.active_set = None
        self.inp_features = None
        self.out_features = None
        self.basis_constraint = 1
        self.res_lbd = None
        self.max_iter=100
        self.dict_max_iter=300
    def flush_variables(self):
        self.active_set = None
        self.inp_features = None
        self.out_features = None

    def initialise_vars(self, feat_dim, des_dim, init_bmat):
        '''
        Initialises B dictionary and s
        '''
        if self.bmat is None:
            if init_bmat is not None:
                if init_bmat.shape[0] == feat_dim and init_bmat.shape[1] == des_dim:
                    self.bmat = init_bmat
                else:
                    raise('Wrong input of initial B matrix, the dimensions' +
                          'should be ' + feat_dim + 'x' + des_dim)
            else:
                self.bmat = random.random((feat_dim, des_dim)) - 0.5
                self.bmat -= mean(self.bmat, axis=0)
                sigm = npsum(self.bmat * self.bmat, axis=0)
                sigm[sigm == 0] = 1
                self.bmat /= sigm
        if self.out_features is None:
            self.out_features = zeros((des_dim, 1))
            self.active_set = zeros((des_dim), bool)
            self.theta = zeros_like(self.out_features)

    def object_val_calc(self, bmat, ksi, gamma, theta, s_old, s_new, _t_):
        bs_new = np.matmul(bmat,s_new)
        bs_old = np.matmul(bmat,s_old)
        reconst_error=(dot(ksi.T, ksi) +
               (1 - _t_)**2 * dot(bs_old.T, bs_old) -
               2 * (1 - _t_) * dot(ksi.T,bs_old) +
               _t_**2 * dot(bs_new.T, bs_new) -
               2 * _t_ * dot(ksi.T,bs_new) +
               2*_t_*(1-_t_)*np.dot(bs_new.T,bs_old))
        res = (reconst_error+gamma * dot(theta.T, _t_ * s_new
                                         + (1 - _t_) * s_old))
        return res,reconst_error

    def feature_sign_search_algorithm(self, inp_features, des_dim,
                                      dist_sigma=1, dist_beta=1, init_bmat=None):
        '''
        Returns sparse features representation
        '''
        self.inp_features=inp_features.copy()
        feat_dim = inp_features.shape[0]
        # Step 1
        self.initialise_vars(feat_dim, des_dim, init_bmat)
        btb = dot(self.bmat.T, self.bmat)
        btf = dot(self.bmat.T, self.inp_features)
        gamma = 2 * dist_sigma**2 * dist_beta
        # Set max iterations
        step2 = 1
        for count in range(self.max_iter):
            # Step 2
            if step2:
                zero_coeffs = (self.out_features == 0)
                qp_der_outfeati = 2 * \
                    (dot(btb, self.out_features) - btf) * zero_coeffs
                i = argmax(npabs(qp_der_outfeati))
                if npabs(qp_der_outfeati[i]) > gamma:
                    self.theta[i] = -sign(qp_der_outfeati[i])
                    self.active_set[i] = True
            # Step 3
            bmat_h = self.bmat[:, self.active_set]
            out_feat_h = self.out_features[self.active_set]
            theta_h = self.theta[self.active_set]
            new_out_f_h = dot(inv(dot(bmat_h.T, bmat_h)), dot(bmat_h.T,
                                                              self.inp_features) -
                              gamma * theta_h / 2.0)
            zero_points_lin_par = out_feat_h / (out_feat_h -
                                                new_out_f_h).astype(float)
            zero_points_lin_par = concatenate((zero_points_lin_par[
                ((zero_points_lin_par > 0)*
                 (zero_points_lin_par < 1)).astype(bool)][:], array([1])), axis=0)
            _t_ = zero_points_lin_par
            objvals,reconst_error = self.object_val_calc(bmat_h, self.inp_features, gamma,
                                           theta_h, out_feat_h,
                                           new_out_f_h, _t_)
            objval_argmin = argmin(objvals)
            new_out_f_h = _t_[objval_argmin] * new_out_f_h + \
                (1 - _t_[objval_argmin]) * out_feat_h
            self.out_features[self.active_set] = new_out_f_h[:]
            self.active_set[self.active_set] = new_out_f_h != 0
            self.theta[self.active_set] = sign(new_out_f_h[
                new_out_f_h != 0])[:, None]
            # Step 4
            nnz_coeff = self.out_features != 0
            # a
            new_qp_der_outfeati = 2 * \
                (dot(btb, self.out_features) - btf)
            cond_a = (new_qp_der_outfeati +
                      gamma * sign(self.out_features)) * nnz_coeff
            if allclose(cond_a, 0):
                # go to cond b:
                z_coeff = self.out_features == 0
                cond_b = npabs(new_qp_der_outfeati * z_coeff) <= gamma
                if npsum(cond_b) == new_qp_der_outfeati.shape[0]:
                    '''
                    print 'Reconstrunction error after '+\
                            'output vector correction',0
                    '''
                    return 0
                else:
                    # go to step 2
                    step2 = 1
            else:
                # go to step 3
                step2 = 0
        '''
        print 'Reconstruction error after output vector correction:',\
                reconst_error.ravel()[objval_argmin]
        '''
        return np.sqrt(reconst_error.ravel()[objval_argmin])

    def lagrange_dual(self,lbd,ksi,_s_):
        #if lbd.shape[0]==1:
        #    lbd=lbd.flatten()
        ksi=self.inp_features
        ksist=dot(ksi,_s_.T)
        #lbd=lbd.flatten()
        try:
            interm_result=inv(dot(_s_,_s_.T)+diag(lbd))
        except np.linalg.linalg.LinAlgError:
            '''
            print 'Singularity met inside LD'
            print '\t sum(lbds)=',npsum(lbd)
            print '\t trace(dot(_s_,_s_.T))=',trace(dot(_s_,_s_.T))
            '''
            interm_result=inv(dot(_s_,_s_.T)+
                              diag(lbd)+
                              0.01*self.basis_constraint*
                              np.eye(lbd.shape[0]))
        res= (dot(dot(ksist,interm_result),ksist.T).trace()+
              (self.basis_constraint*diag(lbd)).trace())
        '''
        print 'Results'
        print '\tF=',res
        print '\tPart1=',dot(dot(ksist,interm_result),ksist.T).trace()
        print '\tPart2=',(self.basis_constraint*diag(lbd)).trace()
        '''
        return res

    def lagrange_dual_grad(self,lbds,ksi,_s_):
        #lbds=lbds.flatten()
        try:
            interm_result=dot(dot(ksi,_s_.T),
                          inv(dot(_s_,_s_.T)+diag(lbds)))
        except np.linalg.linalg.LinAlgError:
            '''
            print 'Singularity met inside LDG'
            print '\t sum(lbds)=',npsum(lbds)
            print '\t trace(dot(_s_,_s_.T))=',trace(dot(_s_,_s_.T))
            '''
            interm_result=dot(dot(ksi,_s_.T),
                          inv(dot(_s_,_s_.T)+
                              diag(lbds)+
                              0.01*self.basis_constraint*
                              np.eye(lbds.shape[0])))
        res=zeros_like(lbds)
        for count in range(res.shape[0]):
            res[count]=-(np.dot(interm_result[:,count].T,
                              interm_result[:,count])-
                        self.basis_constraint)
        return res.reshape(lbds.shape)
    def lagrange_dual_hess(self,lbds,ksi,_s_):
        ksist=dot(ksi,_s_.T)
        #lbds=lbds.flatten()
        try:
            interm_result0=inv(dot(_s_,_s_.T)+diag(lbds))
        except np.linalg.linalg.LinAlgError:
            print 'lagrange_dual_hess'
            print '\tSingularity met, adding eye to inverting part'
            interm_result0=inv(dot(_s_,_s_.T)+diag(lbds)+np.eye(lbds.shape[0]))
        interm_result1=dot(interm_result0,ksist.T)
        res=2*dot(interm_result1,interm_result1.T)*interm_result0
        return res

    def dictionary_training(self):
        prev_err=(self.inp_features-
                       dot(self.bmat,self.out_features))
        prev_err=np.sqrt(trace(dot(prev_err.T,prev_err)))
        '''
        print 'Reconstruction error before dictionary training:',\
                np.sqrt(trace(dot(prev_err.T,prev_err)))
        '''
        minimize_res=fmin_tnc(self.lagrange_dual,
                              (np.ones(self.out_features.shape[0]
                                       ,np.float64)).tolist() ,
                              fprime=self.lagrange_dual_grad,
                              bounds=np.array(([(0,None)]*
                                      self.out_features.shape[0])),
                              disp=0,
                              maxfun=self.dict_max_iter,
                        args=(self.inp_features.copy(),self.out_features.copy()),
                        )
        self.res_lbd=minimize_res[0]
        try:
            self.bmat=dot(dot(self.inp_features,self.out_features.T),
                          inv(dot(self.out_features,
                                  self.out_features.T)+diag(self.res_lbd)))
        except np.linalg.linalg.LinAlgError:
            self.bmat=dot(dot(self.inp_features,self.out_features.T),
                          pinv(dot(self.out_features,
                                  self.out_features.T)+diag(self.res_lbd)))
        reconst_error_mat=self.inp_features-dot(self.bmat,self.out_features)
        reconst_error=np.sqrt(trace(dot(reconst_error_mat.T,
                                        reconst_error_mat)))
        '''
        print 'Reconstruction error after dictionary correction:',\
                reconst_error
        '''
        return prev_err-reconst_error/max(0.1,prev_err)

def main():
    import cv2
    import os.path
    import urllib
    if not os.path.exists('lena.jpg'):
        urllib.urlretrieve('https://www.cosy.sbg.ac'+
                           '.at/~pmeerw/Watermarking/lena_color.gif', 'lena.jpg')
    if not os.path.exists('wolves.jpg'):
        urllib.urlretrieve("https://static.decalgirl.com/assets/designs/large/twolves.jpg",
                           "wolves.jpg")

    test = cv2.imread('lena.jpg', -1)
    test=(test.astype(float))/255.0
    test2 =cv2.imread('wolves.jpg',0)
    test2=test2.astype(float)/255.0
    test = cv2.resize(test, None, fx=0.05, fy=0.05)
    test2 = cv2.resize(test2, test.shape)
    test_shape=test.shape
    spc = FeatureSignSearch()
    inp_features1 = test.ravel()[:, None]
    inp_features2 = test2.ravel()[:, None]
    des_dim = 2 * inp_features1.shape[0]
    dist_sigma = 0.03
    dist_beta = 0.03
    print 'Feature Sign Search Algorithm:'
    reconst_error=spc.feature_sign_search_algorithm(
        inp_features1, des_dim, dist_sigma, dist_beta)
    print '\tLena reconstrunction error:',reconst_error
    out_features1=spc.out_features.copy()
    spc.flush_variables()
    reconst_error=spc.feature_sign_search_algorithm(
        inp_features2, des_dim, dist_sigma, dist_beta)
    print '\tWolves reconstruction error:',reconst_error
    print 'Dictionary Training:'
    out_features2=spc.out_features.copy()
    spc.out_features=concatenate((out_features1,
                                    out_features2),axis=1)
    spc.inp_features=concatenate((inp_features1,
                                    inp_features2),axis=1)
    error=spc.dictionary_training()
    print '\t Final dictionary error:',error
    spc.flush_variables()
    reconst_error=spc.feature_sign_search_algorithm(
        inp_features1, des_dim, dist_sigma, dist_beta)
    print '\tLena reconstruction error:',reconst_error
    reconstructed=(np.dot(spc.bmat,
                              spc.out_features).
                   reshape(test_shape))
    spc.flush_variables()
    reconst_error=spc.feature_sign_search_algorithm(
        inp_features2, des_dim, dist_sigma, dist_beta)
    print '\tWolves reconstruction error:',reconst_error
    reconstructed2=(np.dot(spc.bmat,
                              spc.out_features).
                   reshape(test_shape))

    cv2.imshow('test',test)
    cv2.imshow('reconstructed',reconstructed)
    cv2.imshow('test2',test2)
    cv2.imshow('reconstructed2',reconstructed2)
    cv2.waitKey(0)
if __name__ == '__main__':
    main()
