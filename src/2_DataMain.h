#if !defined(_DataMain_H)
#define _DataMain_H

#include <RcppArmadillo.h>
#include <sstream>      // For convert int to string

class classData {

  public:
  classData() ; ~classData() ; // constructor, destructor
  
  void Initialization() ; 
  
  int p_y, p_x, p_x_star, nu_sigma, nu_phi, n_sample, msg_level, R, S, K  ; 
  double a_R, b_R, a_S, b_S, a_K, b_K, psi_0, b_theta_sq, a_tau, b_tau ; 
  arma::vec D_l_vec, max_R_S_K ; 
  arma::mat Y_NA_mat, X_NA_mat, Phi_0_mat ; 
  
};

class classMain {

  public:
	classMain() ; ~classMain(); // constructor, destructor
	
	void Initialization(classData &Data) ; 
  void Iterate(int Iter, classData &Data) ;
  
  arma::mat Y_mat, X_mat ; 
  std::string where_we_are ; 
  
  // NEWLY ADDED (begin) //
  void Synthesis(classData &Data) ; 
  arma::mat Synt_Y_mat, Synt_X_mat ; 
  // NEWLY ADDED (end) //
  
  
  // To disply in R, moved from private:
  arma::vec r_i_vec, s_i_vec, k_i_vec ; 
  arma::mat test_Y_std_synt ; 
  arma::cube Beta_cube, psi_cube ; 
  
  private:
  classData Data ; 
  
  arma::vec cube_to_vec_fn(arma::cube input_cube, int first_row, int last_row, int first_col, int last_col, int first_slice, int last_slice ) ; 
  arma::vec x_to_x_star_fn(arma::vec x_vec, classData &Data) ; 
  
  double log_dMVN_fn(arma::vec x, arma::vec mu, arma::mat sigma_mat) ;
  double log_dMVN_UT_chol_fn(arma::vec x, arma::vec mu, arma::mat UT_chol) ; 
  arma::vec rMVN_fn(arma::vec mu, arma::mat sigma_mat) ;
  arma::vec rMVN_UT_chol_fn(arma::vec mu, arma::mat UT_chol) ;
  arma::vec rDirichlet_fn( arma::vec alpha_vec ) ; 
  arma::mat rWishart_UT_chol_fn( int nu, arma::mat UT_chol ) ; 
  arma::mat rIW_fn( int nu, arma::mat Mat ) ;
  arma::mat rIW_UT_chol_fn( int nu, arma::mat UT_chol ) ;
  int rdiscrete_fn(arma::vec Prob) ; 
  arma::vec RandVec ; 
  
  void S1_Beta_cube(classData &Data) ;
  void S2_Sigma_cube(classData &Data) ; 
  void S3_theta_mat(classData &Data) ; 
  void S4_Phi_mat(classData &Data) ; 
  void S5_psi_cube(classData &Data) ; 
  void S6a_log_pi_vec(classData &Data) ; 
  void S6b_alpha_K(classData &Data) ; 
  void S6c_k_i_vec(classData &Data) ; 
  void S7a_log_eta_mat(classData &Data) ; 
  void S7b_alpha_R(classData &Data) ; 
  void S7c_r_i_vec(classData &Data) ; 
  void S8a_log_lambda_mat(classData &Data) ; 
  void S8b_alpha_S(classData &Data) ; 
  void S8c_s_i_vec(classData &Data) ; 
  void S9_tau_inv_diag_vec(classData &Data) ; 
  void S_impute_X_mat(classData &Data) ; 
  void S_impute_Y_mat(classData &Data) ; 
  
  double alpha_R, alpha_S, alpha_K ; 
  arma::vec tau_inv_diag_vec, n_r_vec, n_s_vec, n_k_vec ; 
  arma::mat theta_mat, Phi_mat, log_eta_mat, log_lambda_mat, log_pi_vec, n_k_r_mat, n_k_s_mat ; 
  arma::cube UT_Sigma_cube ; 

};


#endif
