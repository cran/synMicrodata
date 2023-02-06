#include "2_DataMain.h"

#define LOG_2_PI 1.83787706640935

//////////// classData /////////////////////

classData::classData(){ }
classData::~classData(){ } //Destructor

void classData::Initialization(){
  
  // n_sample, p_y, p_x is already calculated from Main.Y_mat and Main.X_mat
	// D_l_vec is already read. 
	a_R = 0.5 ; b_R = 0.5 ; a_S = 0.5 ; b_S = 0.5 ; a_K = 1.0 ; b_K = 1.0 ; 
	psi_0 = 1.0 ; b_theta_sq = 10 ; a_tau = 0.5 ; b_tau = 0.5 ; 
	nu_sigma = p_y + 1 ; nu_phi = p_y + 2 ; 
	Phi_0_mat.eye( p_y, p_y ) ; Phi_0_mat = (1.0/(p_y+1)) * Phi_0_mat ;
	p_x_star = sum(D_l_vec) - p_x + 1 ; 
	R = max_R_S_K(0) ; S = max_R_S_K(1) ; K = max_R_S_K(2) ; 
	
} // void classData::Initialization()

//////////// classMain /////////////////////

classMain::classMain(){ }
classMain::~classMain(){ }

void classMain::Initialization(classData &Data){
  
  theta_mat = arma::zeros<arma::mat>(Data.p_y,Data.p_x_star) ; 
  tau_inv_diag_vec = arma::vec(Data.p_x_star) ; tau_inv_diag_vec.fill(1.0) ; 
  Phi_mat = arma::zeros<arma::mat>(Data.p_y,Data.p_y) ; Phi_mat.diag().fill(1.0) ; 
  alpha_R = 1.0 ; alpha_S = 1.0 ; alpha_K = 1.0 ; 
  
  log_eta_mat = arma::mat(Data.K,Data.R) ; log_eta_mat.fill(-1.0 * arma::datum::inf) ; 
  for (int k=0; k<Data.K; k++){
    log_eta_mat.row(k).fill(log( 1.0/Data.R)) ; 
  }
  log_lambda_mat = arma::mat(Data.K,Data.S) ; log_lambda_mat.fill(-1.0 * arma::datum::inf) ; 
  for (int k=0; k<Data.K; k++){
    log_lambda_mat.row(k).fill(log(1.0/Data.S)) ; 
  }
  log_pi_vec = arma::vec(Data.K) ; log_pi_vec.fill(log(1.0/Data.K)) ; 
  
  psi_cube = arma::zeros<arma::cube>(Data.p_x,Data.S,max(Data.D_l_vec)) ; 
  for (int l=0; l<Data.p_x; l++){
    for (int s=0; s<Data.S; s++){  
      psi_cube( arma::span(l,l), arma::span(s,s), arma::span(0,Data.D_l_vec(l)-1) ).fill( 1.0/Data.D_l_vec(l) ) ;
      // std::cout << "l = " << l << ", s = " << s << std::endl ;
      // arma::vec temp_vec = cube_to_vec_fn(psi_cube,l,l,s,s,0,max(Data.D_l_vec)-1) ;
      // std::cout << temp_vec.t() << std::endl ;
    }
  }
  
  Beta_cube = arma::zeros<arma::cube>(Data.p_y,Data.p_x_star,Data.R) ; // Note: for Beta, slice is changing over r
  UT_Sigma_cube = arma::cube(Data.p_y, Data.p_y, Data.R) ; // Note: for Sigma, slice is changing over r
  arma::mat I_mat = arma::zeros<arma::mat>(Data.p_y,Data.p_y) ; I_mat.diag().fill(1.0) ; 
  for (int r=0; r< Data.R; r++){
    UT_Sigma_cube.slice(r) = I_mat ;
    // std::cout << UT_Sigma_cube.slice(r) << std::endl ; 
  }
  
  r_i_vec = arma::zeros<arma::vec>(Data.n_sample) ;
  s_i_vec = arma::zeros<arma::vec>(Data.n_sample) ;
  k_i_vec = arma::zeros<arma::vec>(Data.n_sample) ;
  int init_n_occ_comp = 20 ; 
  if ( Data.R < init_n_occ_comp ) init_n_occ_comp = Data.R ; 
  if ( Data.S < init_n_occ_comp ) init_n_occ_comp = Data.S ; 
  if ( Data.K < init_n_occ_comp ) init_n_occ_comp = Data.K ; 
  arma::vec temp_prob(init_n_occ_comp) ; temp_prob.fill(1.0/init_n_occ_comp) ; // R, S, K should be greater than this number 10. 
  for (int i_sample=0; i_sample<Data.n_sample; i_sample++){
    r_i_vec(i_sample) = rdiscrete_fn(temp_prob) ;
    s_i_vec(i_sample) = rdiscrete_fn(temp_prob) ;
    k_i_vec(i_sample) = rdiscrete_fn(temp_prob) ; 
  }
  // NOTE: For DP, distributed membership indicators are always recommended as starting points. 
  //       Putting same indicators to all mixture components may have local trap issues. 
  
  n_r_vec = arma::zeros<arma::vec>(Data.R) ; n_s_vec = arma::zeros<arma::vec>(Data.S) ; n_k_vec = arma::zeros<arma::vec>(Data.K) ; 
  n_k_r_mat = arma::zeros<arma::mat>(Data.K, Data.R) ;  n_k_s_mat = arma::zeros<arma::mat>(Data.K, Data.S) ;
  for (int i_sample=0; i_sample<Data.n_sample; i_sample++){
    int r_i = r_i_vec(i_sample) ; int s_i = s_i_vec(i_sample) ; int k_i = k_i_vec(i_sample) ; 
    n_r_vec(r_i) = n_r_vec(r_i) + 1 ; n_s_vec(s_i) = n_s_vec(s_i) + 1 ; n_k_vec(k_i) = n_k_vec(k_i) + 1 ; 
    n_k_r_mat(k_i,r_i) = n_k_r_mat(k_i,r_i) + 1 ; n_k_s_mat(k_i,s_i) = n_k_s_mat(k_i,s_i) + 1 ; 
  }
  // std::cout << sum(n_r_vec) << std::endl ; std::cout << sum(n_s_vec) << std::endl ; std::cout << sum(n_k_vec) << std::endl ; 
  
} // void classMain::Initialization

void classMain::Iterate(int Iter, classData &Data) {
	S1_Beta_cube(Data) ;
  S2_Sigma_cube(Data) ;
  S3_theta_mat(Data) ;
  S4_Phi_mat(Data) ;
  S5_psi_cube(Data) ;
  S6a_log_pi_vec(Data) ;
  S6b_alpha_K(Data) ;
  S6c_k_i_vec(Data) ;
  S7a_log_eta_mat(Data) ;
  S7b_alpha_R(Data) ;
  S7c_r_i_vec(Data) ;
  S8a_log_lambda_mat(Data) ;
  S8b_alpha_S(Data) ;
  S8c_s_i_vec(Data) ;
  S9_tau_inv_diag_vec(Data) ;
  S_impute_X_mat(Data) ;
  S_impute_Y_mat(Data) ;
  // std::cout << "Done" << std::endl ;
} 

void classMain::S1_Beta_cube(classData &Data){
  where_we_are = "S1_Beta_cube" ;
  
  arma::mat T_mat = arma::zeros<arma::mat>(Data.p_x_star,Data.p_x_star) ; T_mat.diag() = tau_inv_diag_vec ; 
  
  for (int r=0; r<Data.R; r++){
    
    arma::mat Sigma_r = UT_Sigma_cube.slice(r).t() * UT_Sigma_cube.slice(r) ;
    arma::mat B_r = Beta_cube.slice(r) ;  int n_r = n_r_vec(r) ; 
      
    for (int j=0; j<Data.p_y; j++){
        
      arma::vec loc_vec = arma::zeros<arma::vec>(Data.p_y) ;  loc_vec(j) = 1 ;
      int n_a = sum(loc_vec) ; int n_b = Data.p_y - n_a ;
      arma::mat Sigma_aa = arma::mat(n_a,n_a) ; arma::mat Sigma_ab = arma::mat(n_a,n_b) ; arma::mat Sigma_bb = arma::mat(n_b,n_b) ;
      arma::mat B_b = arma::mat(n_b,Data.p_x_star) ;
      int count_i_a = 0 ; int count_i_b = 0 ;
      for (int i_var=0; i_var<Data.p_y; i_var++){
        if ( loc_vec(i_var)==1 ){
          int count_j_a = 0 ; int count_j_b = 0 ;
          for (int j_var=0; j_var<Data.p_y; j_var++){
            if ( loc_vec(j_var)==1 ){
              Sigma_aa(count_i_a,count_j_a) = Sigma_r(i_var,j_var) ;
              count_j_a++ ;
            } else {
              Sigma_ab(count_i_a,count_j_b) = Sigma_r(i_var,j_var) ;
              count_j_b++ ;
            } // if (loc_vec(j_var)) else ...
          } // for (j_var)
          count_i_a++ ;
        } else {
          B_b.row(count_i_b) = B_r.row(i_var) ;
          int count_j_b = 0 ;
          for (int j_var=0; j_var<Data.p_y; j_var++){
            if ( loc_vec(j_var)==0 ){
              Sigma_bb(count_i_b,count_j_b) = Sigma_r(i_var,j_var) ;
              count_j_b++ ;
            } // if (loc_vec(j_var))
          } // for (j_var)
          count_i_b++ ;
        } // if (loc_vec==1) else ...
      } // for (i_var)
      arma::mat Sigma_bb_inv = Sigma_bb.i() ;
      
      arma::vec y_rj_star_vec(n_r); arma::mat X_r(Data.p_x_star,n_r) ; 
      int count_i = 0 ; 
      for (int i_sample=0; i_sample<Data.n_sample; i_sample++){
        if (r_i_vec(i_sample)==r){
          
          arma::vec y_i_vec = Y_mat.row(i_sample).t() ; 
          arma::vec x_i_vec = X_mat.row(i_sample).t() ;
          arma::vec x_i_star_vec = x_to_x_star_fn(x_i_vec, Data) ;
          
          double y_a = y_i_vec(j) ; arma::vec y_b = arma::vec(n_b) ; 
          int count_i_b = 0 ;
          for (int i_var=0; i_var<Data.p_y; i_var++){
            if ( loc_vec(i_var)==0 ){
              y_b(count_i_b) = y_i_vec(i_var) ; count_i_b++ ;
            } // if (loc_vec==1) else ...
          } // for (i_var)
          arma::vec mu_i_star ; 
          mu_i_star = Sigma_ab * Sigma_bb_inv * ( y_b - B_b * x_i_star_vec ) ; 
          y_rj_star_vec(count_i) = y_a - mu_i_star(0) ; 
          
          X_r.col(count_i) = x_i_star_vec ; 
          
          count_i++ ; 
          
        } // if (==r)
      } // for (i_sample)

      arma::mat temp_mat = Sigma_ab * Sigma_bb_inv * Sigma_ab.t() ; 
      double sig2_star = Sigma_aa(0,0) - temp_mat(0,0) ; 
      arma::mat Cov_star_inv = (1.0/sig2_star)*X_r*X_r.t() + T_mat ;
      
        // std::cout << "." << std::endl ;
      
	  // arma::mat Cov_star = Cov_star_inv.i() ;
      arma::mat Cov_star ;
	  bool ok1 = inv( Cov_star, Cov_star_inv ) ;
	  if ( ok1==TRUE ){
	  	// std::cout << "ok1" << std::endl ;
        arma::vec temp_vec = (1.0/sig2_star)*X_r*y_rj_star_vec + T_mat*theta_mat.row(j).t() ; 
        arma::vec mean_star = Cov_star * temp_vec ;
        // std::cout << "r: " << r << " j: " << j << std::endl ;
        // std::cout << Cov_star << std::endl ;
      
		  // arma::mat UT_Cov_star = arma::chol(Cov_star) ;
		  arma::mat UT_Cov_star ;
        
        // std::cout << "..." << std::endl ;
  	  bool ok = chol(UT_Cov_star, Cov_star) ;
  	  if (ok==TRUE){
          // std::cout << "ok2" << std::endl ;
  	      arma::vec beta_rj = rMVN_UT_chol_fn( mean_star, UT_Cov_star ) ;      
  	      Beta_cube( arma::span(j,j), arma::span(0,Data.p_x_star-1), arma::span(r,r) ) = beta_rj ; 
  	  }
  	  // else { 
  	  // 	std::cout << "Skip S2_Sigma_cube w/ n.p.d" << std::endl ; 
  	  // } //MODIFIED
	  
	  }
	//   else {
	// 
	// 	  std::cout << "Skip S1_Beta_cube w/ n.p.d (2)" << std::endl ;
	// 
	//   } // if (ok1) //MODIFIED
	  	
    } // for (j)
  } // for (r)
    
} // void classMain::S1_Beta_cube

void classMain::S2_Sigma_cube(classData &Data){
  where_we_are = "S2_Sigma_cube" ;
  
  for (int r=0; r<Data.R; r++){
    
    arma::mat B_r = Beta_cube.slice(r) ;  int n_r = n_r_vec(r) ; 
    arma::mat SS_y_Bx = arma::zeros<arma::mat>(Data.p_y,Data.p_y) ;
    
    for (int i_sample=0; i_sample<Data.n_sample; i_sample++){
      if (r_i_vec(i_sample)==r){
        arma::vec y_i_vec = Y_mat.row(i_sample).t() ; 
        arma::vec x_i_vec = X_mat.row(i_sample).t() ;
        arma::vec x_i_star_vec = x_to_x_star_fn(x_i_vec, Data) ;
        arma::vec y_i_Bx = y_i_vec - B_r * x_i_star_vec ; 
        SS_y_Bx = SS_y_Bx + y_i_Bx * y_i_Bx.t() ; 
      } // if (==r)
    } // for (i_sample)
    
    int nu_star = Data.nu_sigma + n_r ; 
    arma::mat Phi_star = Phi_mat + SS_y_Bx ; 
	
	arma::mat UT_Phi_star ;
	  bool ok = chol(UT_Phi_star, Phi_star) ;
	  if (ok==TRUE){
	      arma::mat Sigma_r = rIW_UT_chol_fn( nu_star, UT_Phi_star ) ;
	      UT_Sigma_cube.slice(r) = arma::chol(Sigma_r) ; 
	  } 
	  // else {
	  // std::cout << "Skip S2_Sigma_cube w/ n.p.d" << std::endl ;
	  // } //MODIFIED
	  
    // arma::mat UT_Phi_star = arma::chol(Phi_star) ;
    
  } // for (r)

} // void classMain::S2_Sigma_cube

void classMain::S3_theta_mat(classData &Data){
  where_we_are = "S3_theta_mat" ;
  
  for (int j=0; j<Data.p_y; j++){
    for (int l=0; l<Data.p_x_star; l++){
      double tau_l = 1.0/tau_inv_diag_vec(l) ; 
      double sum_beta_jrl = 0 ; 
      for (int r=0; r<Data.R; r++){
        sum_beta_jrl = sum_beta_jrl + Beta_cube(j,l,r) ; 
      } 
      double sig2_star = 1.0 / (Data.R*tau_l + 1.0/Data.b_theta_sq)  ; 
      double mean_star = sig2_star * tau_l * sum_beta_jrl ; 
      RandVec = Rcpp::rnorm(1,mean_star,sqrt(sig2_star)) ;
      theta_mat(j,l) = RandVec(0) ; 
    } // for (l)
  } // for (j)
  
} // void classMain::S3_theta_mat

void classMain::S4_Phi_mat(classData &Data){
  where_we_are = "S4_Phi_mat" ;
  
  arma::mat Sum_inv_Sigma_r = arma::zeros<arma::mat>(Data.p_y,Data.p_y) ;
  for (int r=0; r<Data.R; r++){
    arma::mat UT_r = UT_Sigma_cube.slice(r) ;
    arma::mat inv_UT_r = UT_r.i() ; 
    Sum_inv_Sigma_r = Sum_inv_Sigma_r + inv_UT_r * inv_UT_r.t() ; // Sigma^-1 = U^-1 L^-1
  } // for (r)
  int nu_star = Data.R * Data.nu_sigma + Data.nu_phi ; 
  arma::mat V = Sum_inv_Sigma_r + Data.Phi_0_mat.i() ; 
  arma::mat UT = chol(V) ; arma::mat inv_UT = UT.i() ; // U_(V^-1) = L^-1 = (U^-1)^T  
  Phi_mat = rWishart_UT_chol_fn( nu_star, inv_UT.t()  ) ; 

} // void classMain::S4_Phi_mat

void classMain::S5_psi_cube(classData &Data){
  where_we_are = "S5_psi_cube" ;
  
  for (int l=0; l<Data.p_x; l++){
    for (int s=0; s<Data.S; s++){
      arma::vec psi_star_vec(Data.D_l_vec(l)) ; psi_star_vec.fill(Data.psi_0) ;
      for (int i_sample=0; i_sample<Data.n_sample; i_sample++){
        if (s_i_vec(i_sample)==s){
          psi_star_vec(X_mat(i_sample,l)) = psi_star_vec(X_mat(i_sample,l)) + 1 ; 
        }
        arma::vec psi_ls_vec = rDirichlet_fn(psi_star_vec) ; 
        psi_cube( arma::span(l,l), arma::span(s,s), arma::span(0,Data.D_l_vec(l)-1) ) = psi_ls_vec ; 
      } // for (i)
    } // for (s)
  } // for (l)

} // void classMain::S5_psi_cube

void classMain::S6a_log_pi_vec(classData &Data){
  where_we_are = "S6a_log_pi_vec" ;
  
  arma::vec nu_short(Data.K-1) ; 
  nu_short.fill(0.1) ; double Sum_n_m = sum(n_k_vec) ;
  for (int k=0; k<(Data.K-1); k++) {
    double one_tilde = 1.0 + n_k_vec(k) ; 
    Sum_n_m = Sum_n_m - n_k_vec(k) ; 
    // start from Sum_n_m - n_k_vec_1 when k=1
    // ->  Sum_n_m - sum(n_k_vec_1 + n_k_vec_2) when k=2 -> ...
    // Sum_n_m - sum(n_k_vec_1 + ... + n_k_vec(k))) i.e. sum_{g=k+1}^K n_k_vec_g
    double alpha_K_tilde = alpha_K + Sum_n_m ;
    RandVec = Rcpp::rbeta( 1, one_tilde, alpha_K_tilde ) ; 
    nu_short(k) = RandVec(0) ; 
  }
  double Sum_logOneMinusVg = 0.0 ;
  for (int k=0; k<(Data.K-1); k++) {
    log_pi_vec(k) = log(nu_short(k)) + Sum_logOneMinusVg ;
    Sum_logOneMinusVg = Sum_logOneMinusVg + log(1.0 - nu_short(k)) ;
  }
  log_pi_vec(Data.K-1) = Sum_logOneMinusVg ;

} // void classMain::S6a_log_pi_vec()

void classMain::S6b_alpha_K(classData &Data){
  where_we_are = "S6b_alpha_K" ;
  
  double a_tilde = Data.a_K + Data.K - 1.0 ;
  double b_tilde = Data.b_K - log_pi_vec(Data.K-1) ;
  // To avoid zero alpha 
  if ( b_tilde > 10.0 ) b_tilde = 10.0 ; // i.e., let pi_vec(Data.K-1) > exp(-10) in drawing alpha
  // To avoid zero alpha 
  RandVec = Rcpp::rgamma(1, a_tilde, 1.0/b_tilde ) ; // Note that b in Rcpp::rgamma(a,b) is scale, i.e., its mean is ab, NOT a/b
  alpha_K = RandVec(0) ;

} // void classMain::S6b_alpha_K

void classMain::S6c_k_i_vec(classData &Data){
  where_we_are = "S6c_k_i_vec" ;
  
  n_k_vec = arma::zeros<arma::vec>(Data.K) ; 
  n_k_r_mat = arma::zeros<arma::mat>(Data.K, Data.R) ;  n_k_s_mat = arma::zeros<arma::mat>(Data.K, Data.S) ;
  
  for (int i_sample=0; i_sample < Data.n_sample; i_sample++) {
    
    int r_i = r_i_vec(i_sample) ;  int s_i = s_i_vec(i_sample) ;
    arma::vec log_Num(Data.K);
    for (int k=0; k<Data.K; k++) {
      log_Num(k) = log_pi_vec(k) + log_eta_mat(k,r_i) + log_lambda_mat(k,s_i) ; 
    }
    double max_log_Num = log_Num.max() ;
    arma::vec pi_star_unnorm = arma::zeros<arma::vec>(Data.K) ; 
    for (int k=0; k<Data.K; k++) {
      pi_star_unnorm(k) = exp( log_Num(k)-max_log_Num ) ; 
    }
    arma::vec pi_star = (1.0/sum(pi_star_unnorm)) * pi_star_unnorm ;
    k_i_vec(i_sample) = rdiscrete_fn( pi_star );
    
    int k_i = k_i_vec(i_sample) ; 
    n_k_vec(k_i) = n_k_vec(k_i) + 1 ; 
    n_k_r_mat(k_i,r_i) = n_k_r_mat(k_i,r_i) + 1 ; n_k_s_mat(k_i,s_i) = n_k_s_mat(k_i,s_i) + 1 ; 
    
  } // for (int i_sample)
  
} // void classMain::S6c_k_i_vec(classData &Data)

void classMain::S7a_log_eta_mat(classData &Data){
  where_we_are = "S7a_log_eta_mat" ;
  
  for (int k=0; k<Data.K; k++){
      
      arma::vec n_r_given_k_vec = n_k_r_mat.row(k).t() ; 
      arma::vec nu_short = arma::zeros<arma::vec>(Data.R-1) ; 
      double Sum_n_m = sum(n_r_given_k_vec) ;
      for (int r=0; r<(Data.R-1); r++) {
        double one_tilde = 1.0 + n_r_given_k_vec(r) ; 
        Sum_n_m = Sum_n_m - n_r_given_k_vec(r) ; 
        // start from Sum_n_m - n_r_vec_1 when r=1
        // ->  Sum_n_m - sum(n_r_vec_1 + n_r_vec_2) when r=2 -> ...
        // Sum_n_m - sum(n_r_vec_1 + ... + n_r_vec(r))) i.e. sum_{g=r+1}^K n_r_vec_g
        double alpha_R_tilde = alpha_R + Sum_n_m ;
        RandVec = Rcpp::rbeta( 1, one_tilde, alpha_R_tilde ) ; 
        nu_short(r) = RandVec(0) ; 
      }
      double Sum_logOneMinusVg = 0.0 ;
    for (int r=0; r<(Data.R-1); r++) {
        log_eta_mat(k,r) = log(nu_short(r)) + Sum_logOneMinusVg ;
        Sum_logOneMinusVg = Sum_logOneMinusVg + log(1.0 - nu_short(r)) ;
      }
      log_eta_mat(k,Data.R-1) = Sum_logOneMinusVg ;
      
  } // for (k)
  
} // void classMain::S7a_log_pi_vec()

void classMain::S7b_alpha_R(classData &Data){
  where_we_are = "S7b_alpha_R" ;
  
  double a_tilde = Data.a_R ; // + 1.0 ;
  double b_tilde = Data.b_R  ;
  for (int k=0; k<Data.K; k++){
    if (n_k_vec(k)>0){
      a_tilde = a_tilde + (Data.R-1) ; 
      b_tilde = b_tilde - log_eta_mat(k,Data.R-1) ; 
    }
  }
  RandVec = Rcpp::rgamma(1, a_tilde, 1.0/b_tilde ) ; // Note that b in Rcpp::rgamma(a,b) is scale, i.e., its mean is ab, NOT a/b
  alpha_R = RandVec(0) ;
  
} // void classMain::S7b_alpha_R

void classMain::S7c_r_i_vec(classData &Data){
  where_we_are = "S7c_r_i_vec" ;
  
  n_r_vec = arma::zeros<arma::vec>(Data.R) ; 
  n_k_r_mat = arma::zeros<arma::mat>(Data.K, Data.R) ; 
  
  for (int i_sample=0; i_sample < Data.n_sample; i_sample++) {
    
    int k_i = k_i_vec(i_sample) ; 
    arma::vec log_DEN = arma::zeros<arma::vec>(Data.R); 
    arma::vec y_i_vec = Y_mat.row(i_sample).t() ; 
    arma::vec x_i_vec = X_mat.row(i_sample).t() ;
    arma::vec x_i_star_vec = x_to_x_star_fn(x_i_vec, Data) ;
    for (int r=0; r<Data.R; r++) {
      arma::mat B_r = Beta_cube.slice(r) ; 
      log_DEN(r) = log_eta_mat(k_i,r) ; 
      log_DEN(r) = log_DEN(r) + log_dMVN_UT_chol_fn( y_i_vec, B_r*x_i_star_vec, UT_Sigma_cube.slice(r) ) ; 
    }
    double max_log_DEN = log_DEN.max() ;
    arma::vec eta_star_unnorm = arma::zeros<arma::vec>(Data.R) ; 
    for (int r=0; r<Data.R; r++){
      eta_star_unnorm(r) =  exp( log_DEN(r)-max_log_DEN ) ; 
    }
    arma::vec eta_star = (1.0/sum(eta_star_unnorm)) * eta_star_unnorm ;
    r_i_vec(i_sample) = rdiscrete_fn( eta_star );
    
    int r_i = r_i_vec(i_sample) ; 
    n_r_vec(r_i) = n_r_vec(r_i) + 1 ; 
    n_k_r_mat(k_i,r_i) = n_k_r_mat(k_i,r_i) + 1 ; 
    
  } // for (int i_sample)
  
} // void classMain::S7c_r_i_vec(classData &Data)

void classMain::S8a_log_lambda_mat(classData &Data){
  where_we_are = "S8a_log_lambda_mat" ;
  
  for (int k=0; k<Data.K; k++){

    arma::vec n_s_given_k_vec = n_k_s_mat.row(k).t() ; 
    arma::vec nu_short(Data.S-1) ; 
    nu_short.fill(0.1) ; double Sum_n_m = sum(n_s_given_k_vec) ;
    for (int s=0; s<(Data.S-1); s++) {
      double one_tilde = 1.0 + n_s_given_k_vec(s) ; 
      Sum_n_m = Sum_n_m - n_s_given_k_vec(s) ; 
      // start from Sum_n_m - n_r_vec_1 when r=1
      // ->  Sum_n_m - sum(n_r_vec_1 + n_r_vec_2) when r=2 -> ...
      // Sum_n_m - sum(n_r_vec_1 + ... + n_r_vec(r))) i.e. sum_{g=r+1}^K n_r_vec_g
      double alpha_S_tilde = alpha_S + Sum_n_m ;
      RandVec = Rcpp::rbeta( 1, one_tilde, alpha_S_tilde ) ; 
      nu_short(s) = RandVec(0) ; 
    }
    double Sum_logOneMinusVg = 0.0 ;
    for (int s=0; s<(Data.S-1); s++) {
      log_lambda_mat(k,s) = log(nu_short(s)) + Sum_logOneMinusVg ;
      Sum_logOneMinusVg = Sum_logOneMinusVg + log(1.0 - nu_short(s)) ;
    }
    log_lambda_mat(k,Data.S-1) = Sum_logOneMinusVg ;
    
  } // for (k)
  
} // void classMain::S8a_log_pi_vec()

void classMain::S8b_alpha_S(classData &Data){
  where_we_are = "S8b_alpha_S" ;
  
  double a_tilde = Data.a_S + 1.0 ;
  double b_tilde = Data.b_S  ;
  for (int k=0; k<Data.K; k++){
    if (n_k_vec(k)>0){
      a_tilde = a_tilde + (Data.S-1) ; 
      b_tilde = b_tilde - log_lambda_mat(k,Data.S-1) ; 
    }
  }
  RandVec = Rcpp::rgamma(1, a_tilde, 1.0/b_tilde ) ; // Note that b in Rcpp::rgamma(a,b) is scale, i.e., its mean is ab, NOT a/b
  alpha_S = RandVec(0) ;
  
} // void classMain::S8b_alpha_S

void classMain::S8c_s_i_vec(classData &Data){
  where_we_are = "S8c_s_i_vec" ;
  
  n_s_vec = arma::zeros<arma::vec>(Data.S) ; 
  n_k_s_mat = arma::zeros<arma::mat>(Data.K, Data.S) ; 
  
  for (int i_sample=0; i_sample < Data.n_sample; i_sample++) {
    
    int k_i = k_i_vec(i_sample) ; 
    arma::vec log_DEN(Data.S); 
    arma::vec x_i_vec = X_mat.row(i_sample).t() ;
    for (int s=0; s<Data.S; s++) {
      log_DEN(s) = log_lambda_mat(k_i,s) ; 
      for (int l=0; l<Data.p_x; l++){
        int x_il = x_i_vec(l) ; // 0 ~ D_l_vec(l)-1 ; 
        log_DEN(s) = log_DEN(s) + log(psi_cube(l,s,x_il)) ; 
      }
    }
    double max_log_DEN = log_DEN.max() ;
    arma::vec lambda_star_unnorm = arma::zeros<arma::vec>(Data.S) ; 
    for (int s=0; s<Data.S; s++){
      lambda_star_unnorm(s) =  exp( log_DEN(s)-max_log_DEN ) ; 
    }
    arma::vec lambda_star = (1.0/sum(lambda_star_unnorm)) * lambda_star_unnorm ;
    s_i_vec(i_sample) = rdiscrete_fn( lambda_star );
    
    int s_i = s_i_vec(i_sample) ; 
    n_s_vec(s_i) = n_s_vec(s_i) + 1 ; 
    n_k_s_mat(k_i,s_i) = n_k_s_mat(k_i,s_i) + 1 ; 
    
  } // for (int i_sample)
  
} // void classMain::S8c_s_i_vec(classData &Data)

void classMain::S9_tau_inv_diag_vec(classData &Data){
  where_we_are = "S9_tau_inv_diag_vec" ;
  
  for (int l=0; l<Data.p_x_star; l++){
    
    double SS_b_theta = 0.0 ; 
    for (int j=0; j<Data.p_y; j++){
      for (int r=0; r<Data.R; r++){
        double dev_b_theta = Beta_cube(j,l,r) - theta_mat(j,l) ; 
        SS_b_theta = SS_b_theta + dev_b_theta*dev_b_theta ; 
      }
    }
    double a_star = Data.a_tau + 0.5 * Data.p_y * Data.R ; 
    double b_star = Data.b_tau + 0.5 * SS_b_theta ; 
    RandVec = Rcpp::rgamma(1, a_star, 1.0/b_star ) ; 
    tau_inv_diag_vec(l) = 1.0/RandVec(0) ; 
    
  } // for (l)
  
} // void classMain::S8c_s_i_vec(classData &Data)

void classMain::S_impute_X_mat(classData &Data){
  where_we_are = "S_impute_X_mat" ;
  
  for (int i_sample=0; i_sample<Data.n_sample; i_sample++){
    // if (Data.X_NA_mat(i_sample,l)==1) // 
    arma::vec y_i_vec = Y_mat.row(i_sample).t() ;
    arma::vec x_i_vec = X_mat.row(i_sample).t() ;
    int r_i = r_i_vec(i_sample) ; int s_i = s_i_vec(i_sample) ; 
    arma::mat B_r_i = Beta_cube.slice(r_i) ; 
    arma::mat UT_Sigma_r_i = UT_Sigma_cube.slice(r_i) ; 
    
    for (int l=0; l<Data.p_x; l++){
      if (Data.X_NA_mat(i_sample,l)==1){
        arma::vec log_DEN = arma::zeros<arma::vec>(Data.D_l_vec(l)) ;  
        for (int i_d=0; i_d<Data.D_l_vec(l); i_d++){
          arma::vec x_i_prop = x_i_vec ; x_i_prop(l) = i_d ; 
          arma::vec x_i_star_prop = x_to_x_star_fn(x_i_prop, Data) ;
          log_DEN(i_d) = log(psi_cube(l,s_i,i_d)) + log_dMVN_UT_chol_fn(y_i_vec, B_r_i*x_i_star_prop, UT_Sigma_r_i) ; 
        }
        double max_log_DEN = log_DEN.max() ;
        arma::vec psi_star_unnorm = arma::zeros<arma::vec>(Data.D_l_vec(l)) ; 
        for (int i_d=0; i_d<Data.D_l_vec(l); i_d++){
          psi_star_unnorm(i_d) =  exp( log_DEN(i_d)-max_log_DEN ) ; 
        }
        arma::vec psi_star = (1.0/sum(psi_star_unnorm)) * psi_star_unnorm ;
        X_mat(i_sample,l) = rdiscrete_fn( psi_star );
      } // if (NA)
    } // for (l)
    
  } // for (int i_sample)
  
} // classMain::S_impute_X_mat

void classMain::S_impute_Y_mat(classData &Data){
  where_we_are = "S_impute_Y_mat" ;
  
  for (int i_sample=0; i_sample<Data.n_sample; i_sample++){
    
    arma::vec s_i = Data.Y_NA_mat.row(i_sample).t() ;
    
    if ( sum(s_i)>0 ){
      
      arma::vec y_i_vec = Y_mat.row(i_sample).t() ;
      arma::vec x_i_vec = X_mat.row(i_sample).t() ;
      arma::vec x_i_star_vec = x_to_x_star_fn(x_i_vec, Data) ;
      int r_i = r_i_vec(i_sample) ; 
      arma::mat B_r_i = Beta_cube.slice(r_i) ; 
      arma::vec mu_i = B_r_i * x_i_star_vec ; 
      arma::mat UT_Sigma_r_i = UT_Sigma_cube.slice(r_i) ; 
      arma::mat Sigma_i = UT_Sigma_r_i.t()*UT_Sigma_r_i ;

      int n_a = sum(s_i) ; int n_b = Data.p_y - n_a ;
      arma::vec mu_a = arma::vec(n_a) ; arma::vec mu_b = arma::vec(n_b) ; arma::vec y_b = arma::vec(n_b) ;
      arma::mat Sigma_aa = arma::mat(n_a,n_a) ; arma::mat Sigma_ab = arma::mat(n_a,n_b) ; arma::mat Sigma_bb = arma::mat(n_b,n_b) ;
      
      int count_i_a = 0 ; int count_i_b = 0 ;
      for (int i_var=0; i_var<Data.p_y; i_var++){
        if ( s_i(i_var)==1 ){
          mu_a(count_i_a) = mu_i(i_var) ;
          int count_j_a = 0 ; int count_j_b = 0 ;
          for (int j_var=0; j_var<Data.p_y; j_var++){
            if ( s_i(j_var)==1 ){
              Sigma_aa(count_i_a,count_j_a) = Sigma_i(i_var,j_var) ;
              count_j_a++ ;
            } else {
              Sigma_ab(count_i_a,count_j_b) = Sigma_i(i_var,j_var) ;
              count_j_b++ ;
            } // if (s_i(j_var)) else ...
          } // for (j_var)
          count_i_a++ ;
        } else {
          mu_b(count_i_b) = mu_i(i_var) ; y_b(count_i_b) = y_i_vec(i_var) ;
          int count_j_b = 0 ;
          for (int j_var=0; j_var<Data.p_y; j_var++){
            if ( s_i(j_var)==0 ){
              Sigma_bb(count_i_b,count_j_b) = Sigma_i(i_var,j_var) ;
              count_j_b++ ;
            } // if (s_i(j_var))
          } // for (j_var)
          count_i_b++ ;
        } // if (s_i==1) else ...
      } // for (i_var)
      
      arma::mat Sigma_bb_inv = Sigma_bb.i() ;
      arma::vec mu_a_star = mu_a + Sigma_ab * Sigma_bb_inv * (y_b-mu_b) ;
      arma::mat Sigma_a_star = Sigma_aa - Sigma_ab * Sigma_bb_inv * Sigma_ab.t() ;
      arma::mat UT_chol_a = arma::chol(Sigma_a_star) ;
      arma::vec y_a = rMVN_UT_chol_fn( mu_a_star, UT_chol_a ) ;
      arma::vec y_i_q = y_i_vec ; 
      count_i_a = 0 ;
      for (int i_var=0; i_var<Data.p_y; i_var++){
        if ( s_i(i_var)==1 ){
          y_i_q(i_var) =  y_a(count_i_a) ;
          count_i_a++ ;
        } // if (s_i==1)
      } // for(i_var)
      Y_mat.row(i_sample) =  y_i_q.t() ;

    } // if ( sum(s_i)>0 )
    
  } // for (int i_sample)
  
} // classMain::S_impute_Y_mat


// NEWLY ADDED (begin) //
void classMain::Synthesis(classData &Data){
  where_we_are = "Synthesis" ;
  Synt_Y_mat = Y_mat ; Synt_X_mat = X_mat ;
  
  for(int i_sample=0; i_sample<Data.n_sample; i_sample++){
  // for(int i_sample=0; i_sample<2; i_sample++){
    
    int s_i = s_i_vec(i_sample); int r_i = r_i_vec(i_sample); // components

    // parameters
    arma::mat B_r_i = Beta_cube.slice(r_i) ; // for continuous variables
    arma::mat UT_Sigma_r_i = UT_Sigma_cube.slice(r_i) ; // for continuous variables

    // Rcpp::Rcout << i_sample << std::endl;
    // Rcpp::Rcout << Data.p_x << std::endl;

    // generate categorical variables
    for(int l=0; l<Data.p_x; l++){ // for 'l'th categorical variables...
      arma::vec psi_s_il = psi_cube.subcube(l,s_i,0,l,s_i,Data.D_l_vec(l)-1);
      Synt_X_mat(i_sample, l) = rdiscrete_fn( psi_s_il );
      
      // Rcpp::Rcout << psi_s_il << std::endl;
      // Rcpp::Rcout << l <<"-th variables: " << Synt_X_mat(i_sample, l) << std::endl;
    }// for (l)

    // x to x_star
    arma::vec x_i_vec = Synt_X_mat.row(i_sample).t() ;
    arma::vec x_i_star_vec = x_to_x_star_fn(x_i_vec, Data) ;

    // generate continuous variables
    Synt_Y_mat.row(i_sample) = rMVN_UT_chol_fn( B_r_i*x_i_star_vec,  UT_Sigma_r_i ).t();
    // Rcpp::Rcout << i_sample <<"-th Y vec : " << Synt_Y_mat.row(i_sample) << std::endl;
  } // for (i)
  
} // classMain::S_synthesis
// NEWLY ADDED (end) //

//////////////////////////////////////
// Functions 

arma::vec classMain::cube_to_vec_fn(arma::cube input_cube, int first_row, int last_row, int first_col, int last_col, int first_slice, int last_slice ){
  int diff_row = last_row - first_row ;
  int diff_col = last_col - first_col ;
  int diff_slice = last_slice - first_slice ;
  if ( ( diff_row < 0 ) || ( diff_col < 0 ) || ( diff_slice < 0 ) ) Rcpp::stop("Incorrect input in classMain::cube_to_vec_fn") ; 
  if ( ( diff_row > 0 ) && ( diff_col > 0 ) ) Rcpp::stop("Incorrect input in classMain::cube_to_vec_fn") ; 
  if ( ( diff_row > 0 ) && ( diff_slice > 0 ) ) Rcpp::stop("Incorrect input in classMain::cube_to_vec_fn") ; 
  if ( ( diff_col > 0 ) && ( diff_slice > 0 ) ) Rcpp::stop("Incorrect input in classMain::cube_to_vec_fn") ; 
  int size_output = diff_row ; 
  if (size_output < diff_col) size_output = diff_col; 
  if (size_output < diff_slice) size_output = diff_slice; 
  size_output = size_output + 1 ; 
  arma::vec output_vec(size_output) ; 
  int count_temp = 0 ; 
  for (int i_row=first_row; i_row<=last_row; i_row++){
    for (int i_col=first_col; i_col<=last_col; i_col++){
      for (int i_slice=first_slice; i_slice<=last_slice; i_slice++){
        output_vec(count_temp) = input_cube(i_row,i_col,i_slice) ; 
        count_temp++ ; 
      }
    }
  }
  return output_vec ; 
} // classMain::cube_to_vec_fn

arma::vec classMain::x_to_x_star_fn(arma::vec x_vec, classData &Data){
  arma::vec x_star = arma::zeros<arma::vec>(Data.p_x_star) ; 
  x_star(0) = 1 ; 
  int prev_end_point = 0 ; 
  for (int l=0; l<Data.p_x; l++){
    if (x_vec(l)>0) x_star(prev_end_point+x_vec(l)) = 1 ; 
    prev_end_point = prev_end_point + Data.D_l_vec(l) - 1 ; 
  } // for (l)
  return x_star ; 
} // classMain::cube_to_vec_fn


// ////////////////////////////////////
// For Rcpp Armadillo
//
// .i() -> inverse of square matrix, same with inv(A) : check sqaure and singular
//  inv(A) -> same with .i() : if use B = inv(A), singluar case returns a bool set to false
//  inv( diagmat(A) ) -> if A is diagonal matrix
//  inv_sympd( A ) -> inverse of symmetric, pd matrix : not check symmetric
//  to solve a system of linear equations, such as Z = inv(X)*Y, using solve() is faster and more accurate
//
//  A.t();    // equivalent to trans(A), but more compact
//
//  psi_cube( arma::span(l,l), arma::span(s,s), arma::span(0,Data.D_l_vec(l)-1) ).fill( 1.0 ) ;
//
//  ** using Rcpp::runif, Rcpp::rbeta, Rcpp::rgamma, Rcpp::rnorm
//      // using R::runif in Rcpp is sometimes unstable
//
//  RandVec = Rcpp::rgamma(1, a, 1.0/b ) ; // for Gamma(a,b) with mean of a/b
//  // Note that b in Rcpp::rgamma(a,b) is scale, i.e., its mean is ab, NOT a/b
//
//  cout::std << " ddd " << cout::endl ;
//  Rprintf("  \n  ") ;
//
//  where_we_are.append("here: ") ;   where_we_are.append( std::to_string( i ) ) ;

//////////////////////////////////////
// Hang Kim's Distribution

double classMain::log_dMVN_fn(arma::vec x, arma::vec mu, arma::mat sigma_mat){
  arma::mat UT_chol = arma::chol(sigma_mat) ;
  return(log_dMVN_UT_chol_fn(x, mu, UT_chol));
  // This function is checked with "dmvnorm" in mvtnorm package on 2018/01/26
}

double classMain::log_dMVN_UT_chol_fn(arma::vec x, arma::vec mu, arma::mat UT_chol){
  arma::mat inv_LTchol = UT_chol.i().t() ; // arma::trans( arma::inv( UT_chol )) ;
  int xdim = x.n_rows;
  double constants = -(xdim/2) * std::log(2.0 * M_PI);
  double sum_log_inv_LTchol = arma::sum(log(inv_LTchol.diag()));
  arma::vec z = inv_LTchol * ( x - mu ) ;
  double logout = constants + sum_log_inv_LTchol - 0.5 * arma::sum(z%z) ; 
  // %	Schur product: element-wise multiplication of two objects 
  // i.e. arma::cum(z%z) = z.t() * z = sum_i z_i^2  // z.t() * z  produces 1 by 1 matrix, so need another line 
  return(logout);
  // This function is checked with "dmvnorm" in mvtnorm package on 2018/01/26
} // arma::vec dmvnrm_arma_mc

arma::vec classMain::rMVN_fn(arma::vec mu_vec, arma::mat Sigma_mat){
  arma::mat UT_chol = arma::chol(Sigma_mat) ;
  return(rMVN_UT_chol_fn(mu_vec, UT_chol)) ;
} // arma::vec classMain::rMVN_fn

arma::vec classMain::rMVN_UT_chol_fn(arma::vec mu, arma::mat UT_chol){
  int n_var = UT_chol.n_rows ; 
  RandVec = Rcpp::rnorm(n_var,0,1) ; 
  arma::mat LT_chol = UT_chol.t() ; 
  arma::vec out = mu + LT_chol * RandVec ; 
  // MVN // y_vec = a_vec + A * z_vec where Sigma = A A^T 
  // Cholesky // Sigma = L L^T = U^T U // arma::chol(Sigma) = U
  return out ;
} // arma::mat rMVN_fn

arma::vec classMain::rDirichlet_fn( arma::vec alpha_vec ){
  int p = alpha_vec.n_rows ; arma::vec y_vec(p) ; 
  for (int j=0; j<p; j++){
    RandVec = Rcpp::rgamma(1, alpha_vec(j), 1.0 ) ; // Gamma(a_j,1)
    y_vec(j) = RandVec(0) ; 
  }
  double sum_y_vec = sum(y_vec) ; 
  arma::vec output_vec = (1.0/sum_y_vec) * y_vec ; 
  return output_vec ; 
} // arma::mat classMain::rDirichlet_fn

arma::mat classMain::rWishart_UT_chol_fn( int nu, arma::mat UT_chol ){
  int p = UT_chol.n_rows ; 
  arma::mat S_mat = arma::zeros<arma::mat>(p,p) ; 
  for (int l=0; l<nu; l++){
    RandVec = Rcpp::rnorm(p,0,1) ; 
    arma::vec x_l = UT_chol.t() * RandVec ;
    S_mat = S_mat + x_l * x_l.t() ; 
  }
  return S_mat ; 
} // arma::mat classMain::rWishart_UT_chol_fn

arma::mat classMain::rIW_fn( int nu, arma::mat Mat ){
  arma::mat UT_chol = arma::chol(Mat) ;
  return rIW_UT_chol_fn(nu, UT_chol) ;
} // arma::mat classMain::rIW_w_pd_check_fn

arma::mat classMain::rIW_UT_chol_fn( int nu, arma::mat UT_chol ){
  
  // Draw IW( nu, UT_chol.t() * UT_chol ) // See 37_Matrix_RandomNumber_Rcpp.pdf ; Function checked 
  
  int p = UT_chol.n_rows ; 
  arma::mat inv_UT_chol = UT_chol.i() ;  
  arma::mat U_mat = arma::zeros<arma::mat>(p,p) ;
  for (int l=0; l<nu; l++){
    RandVec = Rcpp::rnorm(p,0,1) ;       		 
    arma::vec x_l = inv_UT_chol * RandVec ;  
    U_mat = U_mat + x_l * x_l.t() ;										 
  }
  
  arma::mat V_mat = U_mat.i() ; 	
  
  // May result in non-positive definite matrix due to a decimal rounding error
  arma::vec diag_lambda_vec; arma::mat Q_mat;
  eig_sym(diag_lambda_vec, Q_mat, V_mat);
  int count_nonpositive = 0 ;
  while ( diag_lambda_vec.min() <= 0 ){
    for (int i_dim=0; i_dim<p; i_dim++){
      if ( diag_lambda_vec(i_dim) <= 1e-7 ) diag_lambda_vec(i_dim) = 1e-7 ;
    }
    V_mat = Q_mat * diagmat(diag_lambda_vec) * Q_mat.t() ;
    eig_sym(diag_lambda_vec, Q_mat, V_mat) ;
    count_nonpositive++ ;
  } // while
  // if (count_nonpositive>1) std::cout << "rIW_UT_chol_fn has nonpositive matrix " <<  count_nonpositive << " times" << std::endl ; //MODIFIED
  
  arma::mat R_temp ; 
  bool chol_success = chol(R_temp, V_mat) ; 
  if (!chol_success){
    eig_sym(diag_lambda_vec, Q_mat, V_mat) ;
    double max_lambda = diag_lambda_vec.max() ; 
    for (int i_dim=0; i_dim<p; i_dim++){
      if ( diag_lambda_vec(i_dim) <= max_lambda * (1e-7) ) diag_lambda_vec(i_dim) = max_lambda * (1e-7) ;
    }
    V_mat = Q_mat * diagmat(diag_lambda_vec) * Q_mat.t() ;
    // std::cout << "rIW_UT_chol_fn uses max_lambda * (1e-7)" << std::endl ; //MODIFIED
  } 
  
  return V_mat ; 
  
} // arma::mat classMain::rIW_UT_chol_fn 

int classMain::rdiscrete_fn(arma::vec Prob){ 
  // generate an integer from 0 to (max_no-1) with Prob 
  if ( fabs( sum(Prob)-1.0 ) > 1e-10 ) {
    // std::cout << "Prob = " << std::endl ; 
    // std::cout << Prob.t() << std::endl ; 
    // std::cout << "sum(Prob) = " << sum(Prob) << std::endl ;
    // std::cout << "sum(Prob) != 1 in rdiscrete_fn" << std::endl ; //MODIFIED
    Rcpp::stop("sum(Prob) != 1 in rdiscrete_fn") ;  
  }
  int n_vec = Prob.n_rows ; 
  
  // For numerical stability, set zero for pi_k < 1e-05, i.e., one out of 100,000
  // CumProb(i) = CumProb(i-1) + Prob(i) ;  and while ( CumProb(out) < RandVec(0) ) out++ ;
  //   may generate a random value with small prob, more often than its probability
  for (int k=0; k<n_vec; k++){
    if (Prob(k)<1e-05) Prob(k)=0 ;
  }
  Prob = (1.0/sum(Prob)) * Prob ;
  
  arma::vec CumProb = Prob ; 
  for (int i=1; i<n_vec; i++) {
    CumProb(i) = CumProb(i-1) + Prob(i) ;  
  }
  RandVec = Rcpp::runif(1,0,1) ; 
  int out = 0 ; 
  while ( CumProb(out) < RandVec(0) ) out++ ;
  return out;
  // This function is checked with "dmvnorm" in mvtnorm package on 2018/01/26
} // rdiscrete_fn(arma::vec Prob)
