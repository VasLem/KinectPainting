import warnings
import numpy as np
from numpy import array, argmax, argmin, concatenate, diag, isclose
from numpy import dot, sign, zeros, zeros_like, random, trace, mean
from numpy import allclose
from numpy.linalg import inv, pinv, matrix_rank, qr
from numpy import sum as npsum
from numpy import abs as npabs
from scipy.optimize import fmin_tnc
import class_objects as co


class SparseCoding(object):

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
        self.allow_big_vals = False
        self.des_dim = 100
        self.dist_sigma = 0.1
        self.dist_beta = 0.005
        self.theta = None

    def flush_variables(self):
        '''
        Empty variables
        '''
        self.active_set = None
        self.theta = None
        self.inp_features = None
        self.out_features = None

    def initialize(self, feat_dim, des_dim,
                   init_bmat=None, flush_variables=False):
        '''
        Initialises B dictionary and s
        '''
        if (self.bmat is None) or flush_variables:
            if init_bmat is not None:
                if init_bmat.shape[0] == feat_dim and init_bmat.shape[1] == des_dim:
                    self.bmat = init_bmat.copy()
                else:
                    raise Exception('Wrong input of initial B matrix, the dimensions' +
                                    ' should be ' + str(feat_dim) + 'x' +
                                    str(des_dim) + ', not ' +
                                    str(init_bmat.shape[0]) + 'x' +
                                    str(init_bmat.shape[1]))
            else:
                self.bmat = random.random((feat_dim, des_dim)) - 0.5
                self.bmat -= mean(self.bmat, axis=1)[:, None]
                sigm = np.sqrt(npsum(self.bmat * self.bmat, axis=0))
                sigm[sigm == 0] = 1
                self.bmat /= sigm
        if (self.out_features is None) or flush_variables:
            self.out_features = zeros((des_dim, 1))
            self.active_set = zeros((des_dim), bool)
            self.theta = zeros_like(self.out_features)

    def object_val_calc(self, bmat, ksi, gamma, theta, vecs):
        '''
        Calculate objective function value
        '''
        _bs_ = np.dot(bmat, vecs)
        square_term = npsum((ksi - _bs_)**2, axis=0)
        res = (square_term + gamma * dot(theta.T, vecs))
        return res

    def feature_sign_search_algorithm(self,
                                      inp_features=None,
                                      des_dim=0,
                                      dist_sigma=0,
                                      dist_beta=0,
                                      init_bmat=None,
                                      acondtol=1e-3,
                                      display_error=False,
                                      max_iter=0):
        '''
        Returns sparse features representation
        '''
        if des_dim == 0:
            des_dim = self.des_dim
        if dist_sigma == 0:
            dist_sigma = self.dist_sigma
        if dist_beta == 0:
            dist_beta = self.dist_beta
        self.inp_features = inp_features.copy()
        feat_dim = inp_features.shape[0]
        # Step 1
        self.initialize(feat_dim, des_dim, init_bmat)
        self.prev_err = np.linalg.norm(self.inp_features)
        btb = dot(self.bmat.T, self.bmat)
        btf = dot(self.bmat.T, self.inp_features)
        gamma = min([2 * self.dist_sigma**2 * self.dist_beta,
                     co.CONST['max_gamma'], np.max(-2 * btf) / 10])
        # Set max iterations
        step2 = 1
        singularity_met = False
        count = 0
        prev_objval = 0
        if max_iter == 0:
            max_iter = self.max_iter
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
                elif count == 0:
                    gamma = 0.8 * npabs(qp_der_outfeati[i])
                    self.theta[i] = -sign(qp_der_outfeati[i])
                    self.active_set[i] = True
            # Step 3
            bmat_h = self.bmat[:, self.active_set]
            out_feat_h = self.out_features[self.active_set]
            theta_h = self.theta[self.active_set]
            _q_ = dot(bmat_h.T, self.inp_features) - gamma * theta_h / 2.0
            bmat_h2 = dot(bmat_h.T, bmat_h)
            try:
                new_out_f_h = np.linalg.solve(bmat_h2, _q_)
                if np.max(np.abs(new_out_f_h)) > 1000:
                    raise np.linalg.linalg.LinAlgError
            except np.linalg.linalg.LinAlgError:
                rank = matrix_rank(bmat_h2)
                col_space = qr(bmat_h2.T)[0][:, :rank]
                dot_prod = np.sum((_q_ * col_space), axis=0)
                sub_projection = dot_prod / \
                    np.sum(col_space * col_space, axis=0)
                projected_q = npsum(sub_projection * col_space, axis=1)
                if np.allclose(projected_q.ravel(), _q_.ravel(), atol=1e-03):
                    invbmat_h2 = pinv(bmat_h2)
                else:  # has not be seen, so I just add small diagonal quantities
                    warnings.warn('Weird singularity met')
                    singularity_met = True
                    invbmat_h2 = inv(bmat_h2 + 0.001 *
                                     np.eye(bmat_h2.shape[0]))
                    if np.max(np.abs(invbmat_h2)) > 1000:
                        raise Exception('Wrongly computed bmat, try to run' +
                                        ' again or change input features')
                new_out_f_h = dot(invbmat_h2, _q_)

            if np.prod(sign(out_feat_h) != sign(new_out_f_h)):
                zero_points_lin_par = out_feat_h / (out_feat_h -
                                                    new_out_f_h).astype(float)
                zero_points_lin_par = concatenate((zero_points_lin_par[
                    ((zero_points_lin_par > 0) *
                     (zero_points_lin_par < 1)).astype(bool)][:], array([1])), axis=0)
                _t_ = zero_points_lin_par
                null_vecs = _t_ * new_out_f_h + (1 - _t_) * out_feat_h
                objvals = self.object_val_calc(bmat_h, self.inp_features, gamma,
                                               theta_h,
                                               null_vecs).flatten()
                objval_argmin = argmin(objvals)
                objval = np.min(objvals)
                new_out_f_h = null_vecs[:, objval_argmin][:, None].copy()
            else:
                objval = self.object_val_calc(bmat_h, self.inp_features, gamma, theta_h,
                                              new_out_f_h)
            self.out_features[self.active_set] = new_out_f_h.copy()
            self.active_set[self.active_set] = np.logical_not(
                isclose(new_out_f_h, 0))
            self.theta = sign(self.out_features)
            # Step 4
            nnz_coeff = self.out_features != 0
            # a

            new_qp_der_outfeati = 2 * (dot(btb, self.out_features) - btf)
            cond_a = (new_qp_der_outfeati +
                      gamma * sign(self.out_features)) * nnz_coeff
            if np.abs(objval) - np.abs(prev_objval) > 100 and not\
                    self.allow_big_vals and not count == 0:
                print 'Current Objective Function value:', np.abs(objval)
                print 'Previous Objective Function value:', np.abs(prev_objval)
                print 'Problem with big values of inv(B^T*B), increase atol' +\
                    ' or set flag allow_big_vals to true (this might cause' +\
                    ' problems)'
                print new_out_f_h
                exit()
            prev_objval = objval
            if allclose(cond_a, 0, atol=acondtol):
                # go to cond b:
                z_coeff = self.out_features == 0
                cond_b = npabs(new_qp_der_outfeati * z_coeff) <= gamma
                if npsum(cond_b) == new_qp_der_outfeati.shape[0]:
                    '''
                    print 'Reconstrunction error after '+\
                            'output vector correction',0
                    '''
                    final_error = np.linalg.norm(
                        self.inp_features -
                        dot(self.bmat, self.out_features))
                    if display_error:
                        print 'Final Error', final_error
                    return final_error, singularity_met
                else:
                    # go to step 2
                    step2 = 1
            else:
                # go to step 3
                step2 = 0
            if display_error:
                if count % 100 == 0:
                    print '\t Epoch:', count
                    print '\t\t Intermediate Error=', np.linalg.norm(
                        self.inp_features - dot(self.bmat, self.out_features))
        if display_error:
            print 'Algorithm did not converge in the given iterations with' +\
                ' error', np.linalg.norm(
                    self.inp_features - dot(self.bmat,
                                            self.out_features)), ', change' +\
                ' tolerance or increase iterations'
        return (np.linalg.norm(self.inp_features - dot(self.bmat,
                                                       self.out_features)),
                singularity_met)

    def lagrange_dual(self, lbd, ksi, _s_):
        '''
        Lagrange dual function for the minimisation problem
        '''
        ksi = self.inp_features
        ksist = dot(ksi, _s_.T)
        self.res_lbd = np.array(lbd)
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
        return res

    def lagrange_dual_grad(self, lbds, ksi, _s_):
        '''
        Gradient of lagrange dual function, w.r.t. _s_
        '''
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

    '''
    def lagrange_dual_hess(self, lbds, ksi, _s_):
        ksist = dot(ksi, _s_.T)
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
    '''
    def dictionary_training(self):
        '''
        print 'Reconstruction error before dictionary training:',\
                np.sqrt(trace(dot(prev_err.T,prev_err)))
        '''
        prev_err_mat = self.inp_features - dot(self.bmat, self.out_features)
        self.prev_err = np.linalg.norm(prev_err_mat)

        fmin_tnc(self.lagrange_dual,
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
    '''
    Example function
    '''
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
    bmat = None
    for count in range(4):
        print 'Iteration', count
        lena_sparse = SparseCoding()
        wolves_sparse = SparseCoding()
        if count > 1:
            init_bmat = bmat.copy()
        else:
            init_bmat = None
        lena_sparse.max_iter = 500
        wolves_sparse.max_iter = 500
        inp_features1 = test.ravel()[:, None]
        inp_features2 = test2.ravel()[:, None]
        lena_sparse.des_dim = 2 * inp_features1.shape[0]
        lena_sparse.dist_sigma = 0.01
        lena_sparse.dist_beta = 0.01
        lena_error = lena_sparse.feature_sign_search_algorithm(
            inp_features1, init_bmat=init_bmat, display_error=False)
        wolves_sparse.des_dim = 2 * inp_features1.shape[0]
        wolves_sparse.dist_sigma = 0.01
        wolves_sparse.dist_beta = 0.01
        wolves_error = wolves_sparse.feature_sign_search_algorithm(
            inp_features2, init_bmat=lena_sparse.bmat.copy(),
            display_error=False)
        if count == 0:
            print 'Sparse features initialisation'
            print '\tLena:Previous Error:', lena_sparse.prev_err
            print '\t\tCurrent Error', lena_error[0]
            print '\tWolves:Previous Error:', wolves_sparse.prev_err
            print '\t\tCurrent Error:', wolves_error[0]
            print '\nDictionary Training:'
        if count == 3:
            lena_reconstructed = dot(
                lena_sparse.bmat, lena_sparse.out_features)
            wolves_reconstructed = dot(
                wolves_sparse.bmat, wolves_sparse.out_features)
            print 'Final Errors'
            print '\tLena:', lena_error[0]
            print '\tWolves:', wolves_error[0]
            print '\nDictionary Training:'

        else:
            out_features1 = lena_sparse.out_features.copy()
            out_features2 = wolves_sparse.out_features.copy()
            dictionary = SparseCoding()
            dictionary.bmat = lena_sparse.bmat.copy()
            dictionary.out_features = concatenate((out_features1,
                                                   out_features2), axis=1)
            dictionary.inp_features = concatenate((inp_features1,
                                                   inp_features2), axis=1)
            dict_err_mat = (dictionary.inp_features -
                            dot(dictionary.bmat, dictionary.out_features))
            dict_err = np.linalg.norm(dict_err_mat)
            if count == 0:
                print '\tInitial Error:', dict_err
            else:
                print '\tError after iteration', count - 1, dict_err
            dictionary.display = 5
            bmat = dictionary.dictionary_training()

    cv2.imshow('test', test)
    cv2.imshow('reconstructed', lena_reconstructed.reshape(test_shape))
    cv2.imshow('test2', test2)
    cv2.imshow('reconstructed2', wolves_reconstructed.reshape(test_shape))
    cv2.waitKey(0)
if __name__ == '__main__':
    main()
