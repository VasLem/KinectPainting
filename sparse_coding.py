import numpy as np
from numpy import array, argmax, argmin, concatenate, diag,isclose
from numpy import dot, sign, zeros, zeros_like, random, trace, mean
from numpy import allclose,cross
from numpy.linalg import inv, pinv, norm, matrix_rank, qr
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
        self.max_iter = 500
        self.dict_max_iter = 300
        self.display = 0
        self.prev_err = 0
        self.curr_error = 0

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
                self.bmat -= mean(self.bmat, axis=1)[:,None]
                sigm = np.sqrt(npsum(self.bmat * self.bmat, axis=0))
                sigm[sigm == 0] = 1
                self.bmat /= sigm
        if self.out_features is None:
            self.out_features = zeros((des_dim, 1))
            self.active_set = zeros((des_dim), bool)
            self.theta = zeros_like(self.out_features)

    def object_val_calc(self, bmat, ksi, gamma, theta, vecs):
        
        _bs_ = np.dot(bmat, vecs)
        square_term = npsum((ksi-_bs_)**2,axis=0)
        res = (square_term + gamma * dot(theta.T, vecs))
        return res

    def feature_sign_search_algorithm(self, inp_features, des_dim,
                                      dist_sigma=1, dist_beta=1,
                                      init_bmat=None,acondtol=1e-3,
                                      display_error=0):
        '''
        Returns sparse features representation
        '''
        self.inp_features = inp_features.copy()
        feat_dim = inp_features.shape[0]
        # Step 1
        self.initialise_vars(feat_dim, des_dim, init_bmat)
        self.prev_err = np.linalg.norm(self.inp_features)
        btb = dot(self.bmat.T, self.bmat)
        btf = dot(self.bmat.T, self.inp_features)
        gamma=min(0.0001,np.max(-2*btf)/10)
        #gamma = 2 * dist_sigma**2 * dist_beta
        # Set max iterations
        step2 = 1
        singularity_met=False
        count=0
        prev_objval=0
        _q_prev=None
        bmat_h_prev=None
        for count in range(self.max_iter):
            # Step 2
            if step2:
                zero_coeffs = (self.out_features == 0)
                qp_der_outfeati = 2 * \
                    (dot(btb, self.out_features) - btf) * zero_coeffs
                i = argmax(npabs(qp_der_outfeati))
                if npabs(qp_der_outfeati[i]) >= gamma:
                    self.theta[i] = -sign(qp_der_outfeati[i])
                    self.active_set[i] = True
                elif count==0:
                    gamma=0.8*npabs(qp_der_outfeati[i])
                    self.theta[i] = -sign(qp_der_outfeati[i])
                    self.active_set[i] = True
            # Step 3
            bmat_h = self.bmat[:, self.active_set]
            out_feat_h = self.out_features[self.active_set]
            theta_h = self.theta[self.active_set]
            _q_=dot(bmat_h.T,self.inp_features) - gamma * theta_h / 2.0
            bmat_h2=dot(bmat_h.T,bmat_h)
            try:
                invbmat_h2=inv(bmat_h2)
                if np.max(np.abs(invbmat_h2))>1000:
                    invbmat_h2=pinv(bmat_h2)
            except np.linalg.linalg.LinAlgError:
                rank=matrix_rank(bmat_h2)
                col_space=qr(bmat_h2.T)[0][:,:rank]
                dot_prod=np.sum((_q_*col_space),axis=0)
                sub_projection=dot_prod/np.sum(col_space*col_space,axis=0)
                projected_q=npsum(sub_projection*col_space,axis=1)
                if np.allclose(projected_q.ravel(),_q_.ravel(),atol=1e-03):
                    invbmat_h2 = pinv(bmat_h2)
                else: #has not be seen, so I just add small diagonal quantities
                    singularity_met=True
                    invbmat_h2 = inv(bmat_h2+0.001*
                                          np.eye(bmat_h2.shape[0]))

            new_out_f_h=dot(invbmat_h2,_q_)
            lol=0
            #lol=np.all(out_feat_h==0)
            if np.prod(sign(out_feat_h)!=sign(new_out_f_h)):
                zero_points_lin_par = out_feat_h / (out_feat_h -
                                                    new_out_f_h).astype(float)
                zero_points_lin_par = concatenate((zero_points_lin_par[
                    ((zero_points_lin_par > 0) *
                     (zero_points_lin_par < 1)).astype(bool)][:], array([1])), axis=0)
                _t_ = zero_points_lin_par
                null_vecs = _t_*new_out_f_h+(1-_t_)*out_feat_h
                objvals= self.object_val_calc(bmat_h, self.inp_features, gamma,
                                                              theta_h,
                                                              null_vecs).flatten()
                objval_argmin = argmin(objvals)
                objval=np.min(objvals)
                new_out_f_h = null_vecs[:,objval_argmin][:,None].copy()
            else:
                objval=self.object_val_calc(bmat_h,self.inp_features,gamma,theta_h,
                                       new_out_f_h)
            self.out_features[self.active_set] = new_out_f_h.copy()
            self.active_set[self.active_set] = np.logical_not(isclose(new_out_f_h
                                                                      , 0))
            self.theta= sign(self.out_features)
            # Step 4 
            nnz_coeff = self.out_features != 0
            # a

            new_qp_der_outfeati = 2 *(dot(btb, self.out_features) - btf)
            cond_a = (new_qp_der_outfeati +
                      gamma * sign(self.out_features)) * nnz_coeff
            if np.abs(objval)-np.abs(prev_objval)>100:
                print 'Problem with bmat singularity, increase atol'
                exit()
            prev_objval=objval
            _q_prev=_q_.copy()
            bmat_h_prev=bmat_h2.copy()
            
            if allclose(cond_a, 0,atol=acondtol):
                # go to cond b:
                z_coeff = self.out_features == 0
                cond_b = npabs(new_qp_der_outfeati * z_coeff) <= gamma
                if npsum(cond_b) == new_qp_der_outfeati.shape[0]:
                    '''
                    print 'Reconstrunction error after '+\
                            'output vector correction',0
                    '''
                    final_error=np.linalg.norm(
                            self.inp_features-
                            dot(self.bmat,self.out_features))
                    if display_error:
                        print 'Final Error',final_error
                    return final_error,singularity_met
                else:
                    # go to step 2
                    step2 = 1
            else:
                # go to step 3
                step2 = 0
            '''
            print 'Error:',np.linalg.norm(
                    self.inp_features-dot(self.bmat,self.out_features))
            '''
            if display_error:
                if count%100==0:
                    print '\t Epoch:',count
                    print '\t\t Intermediate Error=',np.linalg.norm(
                        self.inp_features-dot(self.bmat,self.out_features))
        '''
        print 'Reconstruction error after output vector correction:',\
                reconst_error.ravel()[objval_argmin]
        '''
        print 'Algorithm did not converge in the given iterations with'+\
                    ' error',np.linalg.norm(
                        self.inp_features-dot(self.bmat,
                                              self.out_features)),', change'+\
                        ' tolerance or increase iterations'
        return (np.linalg.norm(self.inp_features-dot(self.bmat,
                                                     self.out_features)),
                singularity_met)

    def lagrange_dual(self, lbd, ksi, _s_):
        # if lbd.shape[0]==1:
        #    lbd=lbd.flatten()
        ksi = self.inp_features
        ksist = dot(ksi, _s_.T)
        self.res_lbd = np.array(lbd)
        # lbd=lbd.flatten()
        try:
            interm_result = inv(dot(_s_, _s_.T) + diag(lbd))
        except np.linalg.linalg.LinAlgError:
            print 'Singularity met inside LD'
            print '\t sum(lbds)=', npsum(lbd)
            print '\t trace(dot(_s_,_s_.T))=', trace(dot(_s_, _s_.T))
            interm_result = inv(dot(_s_, _s_.T) +
                                diag(lbd) +
                                0.01 * self.basis_constraint *
                                np.eye(lbd.shape[0]))
        res = (dot(dot(ksist, interm_result), ksist.T).trace() +
               (self.basis_constraint * diag(lbd)).trace())
        '''
        print 'Results'
        print '\tF=',res
        print '\tPart1=',dot(dot(ksist,interm_result),ksist.T).trace()
        print '\tPart2=',(self.basis_constraint*diag(lbd)).trace()
        '''
        return res

    def lagrange_dual_grad(self, lbds, ksi, _s_):
        # lbds=lbds.flatten()
        try:
            interm_result = dot(dot(ksi, _s_.T),
                                inv(dot(_s_, _s_.T) + diag(lbds)))
        except np.linalg.linalg.LinAlgError:
            print 'Singularity met inside LDG'
            print '\t sum(lbds)=', npsum(lbds)
            print '\t trace(dot(_s_,_s_.T))=', trace(dot(_s_, _s_.T))
            interm_result = dot(dot(ksi, _s_.T),
                                inv(dot(_s_, _s_.T) +
                                    diag(lbds) +
                                    self.basis_constraint *
                                    np.eye(lbds.shape[0])))
        res = zeros_like(lbds)
        for count in range(res.shape[0]):
            res[count] = -(np.dot(interm_result[:, count].T,
                                  interm_result[:, count]) -
                           self.basis_constraint)
        return res.reshape(lbds.shape)

    def lagrange_dual_hess(self, lbds, ksi, _s_):
        ksist = dot(ksi, _s_.T)
        # lbds=lbds.flatten()
        try:
            interm_result0 = inv(dot(_s_, _s_.T) + diag(lbds))
        except np.linalg.linalg.LinAlgError:
            print 'lagrange_dual_hess'
            print '\tSingularity met, adding eye to inverting part'
            interm_result0 = inv(
                dot(_s_, _s_.T) + diag(lbds) + np.eye(lbds.shape[0]))
        interm_result1 = dot(interm_result0, ksist.T)
        res = 2 * dot(interm_result1, interm_result1.T) * interm_result0
        return res

    def dictionary_training(self):
        '''
        print 'Reconstruction error before dictionary training:',\
                np.sqrt(trace(dot(prev_err.T,prev_err)))
        '''
        prev_err_mat = self.inp_features - dot(self.bmat, self.out_features)
        self.prev_err = np.linalg.norm(prev_err_mat)

        minimize_res = fmin_tnc(self.lagrange_dual,
                                (np.ones(self.out_features.shape[
                                 0], np.float64)).tolist(),
                                fprime=self.lagrange_dual_grad,
                                bounds=np.array(([(10**(-18), 10**4)] *
                                                 self.out_features.shape[0])),
                                stepmx=10.0,
                                disp=self.display,
                                args=(self.inp_features.copy(),
                                      self.out_features.copy()))
        try:
            bmat = dot(dot(self.inp_features, self.out_features.T),
                       inv(dot(self.out_features,
                               self.out_features.T) + diag(self.res_lbd)))
        except np.linalg.linalg.LinAlgError:
            print 'singularity in dictionary training'
            bmat = dot(dot(self.inp_features, self.out_features.T),
                       inv(dot(self.out_features,
                               self.out_features.T) +
                           diag(self.res_lbd) +
                           0.01 * self.basis_constraint *
                           np.eye(self.res_lbd.shape[0])))
        '''
        print 'Reconstruction error after dictionary correction:',\
                reconst_error
        '''
        return bmat


def main():
    import cv2
    import os.path
    import urllib
    if not os.path.exists('lena.jpg'):
        urllib.urlretrieve('https://www.cosy.sbg.ac' +
                           '.at/~pmeerw/Watermarking/lena_color.gif', 'lena.jpg')
    if not os.path.exists('wolves.jpg'):
        urllib.urlretrieve("https://static.decalgirl.com/assets/designs/large/twolves.jpg",
                           "wolves.jpg")

    test = cv2.imread('lena.jpg', -1)
    test = (test.astype(float)) / 255.0
    test2 = cv2.imread('wolves.jpg', 0)
    test2 = test2.astype(float) / 255.0
    test = cv2.resize(test, None, fx=0.05, fy=0.05)
    test2 = cv2.resize(test2, test.shape)
    test_shape = test.shape
    spc = FeatureSignSearch()
    inp_features1 = test.ravel()[:, None]
    inp_features2 = test2.ravel()[:, None]
    des_dim = 2 * inp_features1.shape[0]
    dist_sigma = 0.001
    dist_beta = 0.001
    print 'Feature Sign Search Algorithm:'
    reconst_error = spc.feature_sign_search_algorithm(
        inp_features1, des_dim, dist_sigma, dist_beta)
    print '\tLena:Before:', spc.prev_err
    print '\t\tAfter', reconst_error
    out_features1 = spc.out_features.copy()
    spc.flush_variables()
    reconst_error = spc.feature_sign_search_algorithm(
        inp_features2, des_dim, dist_sigma, dist_beta)
    print '\tWolves:Before:', spc.prev_err
    print '\t\tAfter:', reconst_error
    print 'Dictionary Training:'
    out_features2 = spc.out_features.copy()
    spc.out_features = concatenate((out_features1,
                                    out_features2), axis=1)
    spc.inp_features = concatenate((inp_features1,
                                    inp_features2), axis=1)
    prev_err = (spc.inp_features -
                dot(spc.bmat, spc.out_features))
    prev_err = np.sqrt(trace(dot(prev_err.T, prev_err)))
    print '\t\tBefore:', prev_err
    repeat = 1
    spc.display = 5
    while repeat:
        bmat = spc.dictionary_training()
        reconst_error_mat = spc.inp_features - dot(bmat, spc.out_features)
        error = np.sqrt(trace(dot(reconst_error_mat.T,
                                  reconst_error_mat)))
        if error > prev_err:
            print '\t Warning:Dictionary training not completed' +\
                ', must do it again, error=', error
            print '(current_error-previous_error=', error - prev_err, ')'
            repeat = 0
        else:
            repeat = 0
    spc.bmat = bmat.copy()
    print '\t\tAfter:', error
    spc.flush_variables()
    spc.feature_sign_search_algorithm(
        inp_features1, des_dim, dist_sigma, dist_beta)
    reconstructed = np.dot(spc.bmat,
                           spc.out_features)
    reconst_error_mat = inp_features1 - reconstructed
    rec_error = np.sqrt(dot(reconst_error_mat.T, reconst_error_mat).trace())
    print '\tLena:', rec_error
    spc.flush_variables()
    reconst_error = spc.feature_sign_search_algorithm(
        inp_features2, des_dim, dist_sigma, dist_beta)
    reconstructed2 = np.dot(spc.bmat,
                            spc.out_features)
    reconst_error_mat = inp_features2 - reconstructed2
    rec_error = np.sqrt(dot(reconst_error_mat.T, reconst_error_mat).trace())
    print '\tWolves:', rec_error

    cv2.imshow('test', test)
    cv2.imshow('reconstructed', reconstructed.reshape(test_shape))
    cv2.imshow('test2', test2)
    cv2.imshow('reconstructed2', reconstructed2.reshape(test_shape))
    cv2.waitKey(0)
if __name__ == '__main__':
    main()
