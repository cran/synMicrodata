/////////////////////////////////////////////////////////////////////
// Change log
// 
// 2018-05-04, 1.3.0
//  * Add functions in 2_Function_createModel_Run.R to check convergences.
// 
// 2018-05-09, 1.3.1 
//  * Replace ".cube_UT_cholSigma" with ".stacked_UT_cholSigma" 
//    as the old version of RcppArmadillo in REHL5 does not support it.
//    However, cube types can be used for computaitons. 
//
// 2018-05-09, 1.3.2
//  * Prevent alpha = 0 
//   : change 2_DataMain.cpp / S6_pi
//
// 2018-05-17, 1.3.3
//  * Some errors during initialization (with balance edits)
//  * Add a small epsilon 1e-6 to readData function in 1_Function_readData.R 
//  * max( Edit_matrix2[,1:n_var]%*%t(Y_in_checked[i_sample,])-Edit_matrix2[,"CONSTANT"] ) > ( 0 + 1e-6 ) 
//
// 2018-09-17, 1.3.5
//  * Some errors during initialization (with balance edits)
//  * Increase the epsilon to 1e-3 to readData function in 1_Function_readData.R 
//  * max( Edit_matrix2[,1:n_var]%*%t(Y_in_checked[i_sample,])-Edit_matrix2[,"CONSTANT"] ) > ( 0 + 1e-3 ) 
//
// 2018-09-17, 1.4.1
//  * Back to ( 0 + 1e-6 ) in readData function in 1_Function_readData.R
//  * Change function classMain::S_Add_SyntY in 2_DataMain.cpp
//  *   -> a few z_i's (whose y_i's are close to zero) forms a mixture component with a large variance. 
//  *      Do not necessarily use z_i, so redraw k from Categorical(pi) for the (few) units 
/////////////////////////////////////////////////////////////////////

#include "1_IO.h"

///////////////////////
RCPP_MODULE(IO_module){
  
  using namespace R ;
  using namespace Rcpp ;
  
  class_<classIO>( "modelobject" )
    
    .constructor< arma::vec >()     
    
    .property("Y_mat", &classIO::GetY_mat, &classIO::SetY_mat, "Y")
    .property("X_mat", &classIO::GetX_mat, &classIO::SetX_mat, "X")
    .property(".Y_NA_mat", &classIO::GetY_NA_mat, &classIO::SetY_NA_mat, "Y_NA")
    .property(".X_NA_mat", &classIO::GetX_NA_mat, &classIO::SetX_NA_mat, "X_NA")
    .property(".D_l_vec", &classIO::GetD_l_vec, &classIO::SetD_l_vec, ".D_l_vec")
    
    .property("msg_level", &classIO::Getmsg_level, &classIO::Setmsg_level, "0: errors; 1: error and warnings; 2: errors, warnings and info")
    .property(".where_we_are", &classIO::Getwhere_we_are, &classIO::Setwhere_we_are, "where_we_are")
    
    .method(".Initialization", &classIO::Initialization, "Initialization")
    .method(".Iterate", &classIO::Iterate, "Run one iteration of MCMC algorithm")
    .method(".Run", &classIO::Run, "Run multiple iterations of MCMC algorithm")
    
    // To check the code 
    .property("r_i_vec", &classIO::Getr_i_vec, &classIO::Setr_i_vec, "r_i_vec")
    .property("s_i_vec", &classIO::Gets_i_vec, &classIO::Sets_i_vec, "s_i_vec")
    .property("k_i_vec", &classIO::Getk_i_vec, &classIO::Setk_i_vec, "k_i_vec")
    .property(".psi_cube", &classIO::Getpsi_cube, &classIO::Setpsi_cube, "psi_cube")
    .property(".Beta_cube", &classIO::GetBeta_cube, &classIO::SetBeta_cube, "Beta_cube")
    
    // NEWLY ADDED (begin) //
    .method(".Synthesis", &classIO::Synthesis, "Generate synthetic data")
    .property("Synt_Y_mat", &classIO::GetSynt_Y_mat, "GetSynt_Y_mat") 
    .property("Synt_X_mat", &classIO::GetSynt_X_mat, "GetSynt_X_mat") 
    // NEWLY ADDED (end) //
    
    .property(".test_Y_std_synt", &classIO::Gettest_Y_std_synt, &classIO::Settest_Y_std_synt, "test_Y_std_synt")
    
    // To check Hang Kim's function
    // .method(".test_log_dMVN_fn", &classIO::test_log_dMVN_fn, "test_log_dMVN_fn")
    // .method(".test_log_dMVN_UT_chol_fn", &classIO::test_log_dMVN_UT_chol_fn, "test_log_dMVN_UT_chol_fn")
    // .method(".test_rMVN_fn", &classIO::test_rMVN_fn, "test_rMVN_fn")
    // .method(".test_rMVN_UT_chol_fn", &classIO::test_rMVN_UT_chol_fn, "test_rMVN_UT_chol_fn")
    // .method(".test_rIW_fn", &classIO::test_rIW_fn, "test_rIW_fn")
    // .method(".test_rdiscrete_fn", &classIO::test_rdiscrete_fn, "test_rdiscrete_fn")
    
    ; // Do not delete ;
  
}       

///////////////////////
classIO::classIO(arma::vec max_R_S_K_) {
  Data.max_R_S_K = max_R_S_K_ ; 
} // classIO::classIO

classIO::~classIO(){
} //Destructor

void classIO::Initialization() {
  IterCount = 0 ;
  Data.n_sample = Main.Y_mat.n_rows ; Data.p_y = Main.Y_mat.n_cols ; Data.p_x = Main.X_mat.n_cols ; 
  Data.Initialization() ;
  Main.Initialization(Data) ;
}

void classIO::Iterate(){
  IterCount++; // std::cout << "IterCount" << std::endl ;
  Main.Iterate(IterCount, Data) ;
}

void classIO::Run(int n_iter_){
  for (int i_iter=0; i_iter<n_iter_; i_iter++) {
    IterCount++;
    Main.Iterate(IterCount, Data) ;
  }
}

arma::mat classIO::GetY_mat() { return Main.Y_mat ; }
void classIO::SetY_mat(arma::mat Y_mat_) { Main.Y_mat = Y_mat_ ; }

arma::mat classIO::GetX_mat() { return Main.X_mat ; }
void classIO::SetX_mat(arma::mat X_mat_) { Main.X_mat = X_mat_ ; }

arma::mat classIO::GetY_NA_mat() { return Data.Y_NA_mat ; }
void classIO::SetY_NA_mat(arma::mat Y_NA_mat_) { Data.Y_NA_mat = Y_NA_mat_ ; }

arma::mat classIO::GetX_NA_mat() { return Data.X_NA_mat ; }
void classIO::SetX_NA_mat(arma::mat X_NA_mat_) { Data.X_NA_mat = X_NA_mat_ ; }

arma::vec classIO::GetD_l_vec() { return Data.D_l_vec ; }
void classIO::SetD_l_vec(arma::vec D_l_vec_) { Data.D_l_vec = D_l_vec_ ; }

int classIO::Getmsg_level() { return Data.msg_level ; }
void classIO::Setmsg_level(int msg_level_) { Data.msg_level = msg_level_ ; }
std::string classIO::Getwhere_we_are() { return Main.where_we_are ; }
void classIO::Setwhere_we_are(std::string where_we_are_) { Main.where_we_are = where_we_are_ ; }

// NEWLY ADDED (begin) //
void classIO::Synthesis(){
  Main.Synthesis(Data) ;
}
arma::mat classIO::GetSynt_Y_mat() { return Main.Synt_Y_mat ; }
arma::mat classIO::GetSynt_X_mat() { return Main.Synt_X_mat ; }
// NEWLY ADDED (end) // 

///////////////////  To check the code  //////////////////////////////

arma::cube classIO::GetBeta_cube() { return Main.Beta_cube ; }
void classIO::SetBeta_cube(arma::cube Beta_cube_) { Main.Beta_cube = Beta_cube_ ; }
arma::vec classIO::Getr_i_vec() { return Main.r_i_vec ; }
void classIO::Setr_i_vec(arma::vec r_i_vec_) { Main.r_i_vec = r_i_vec_ ; }
arma::vec classIO::Gets_i_vec() { return Main.s_i_vec ; }
void classIO::Sets_i_vec(arma::vec s_i_vec_) { Main.s_i_vec = s_i_vec_ ; }
arma::vec classIO::Getk_i_vec() { return Main.k_i_vec ; }
void classIO::Setk_i_vec(arma::vec k_i_vec_) { Main.k_i_vec = k_i_vec_ ; }
arma::cube classIO::Getpsi_cube() { return Main.psi_cube ; }
void classIO::Setpsi_cube(arma::cube psi_cube_) { Main.psi_cube = psi_cube_ ; }
arma::mat classIO::Gettest_Y_std_synt() { return Main.test_Y_std_synt ; }
void classIO::Settest_Y_std_synt(arma::mat test_Y_std_synt_) { Main.test_Y_std_synt = test_Y_std_synt_ ; }
  
  
///////////////////  To check Hang Kim's function //////////////////////////////

// double classIO::test_log_dMVN_fn(arma::vec x, arma::vec mu, arma::mat sigma_mat){
//   return(Main.log_dMVN_fn(x, mu, sigma_mat)) ;
// } // This function is checked with "dmvnorm" in mvtnorm package on 2018/01/26
// double classIO::test_log_dMVN_UT_chol_fn(arma::vec x, arma::vec mu, arma::mat UT_chol){
//   return( Main.log_dMVN_UT_chol_fn(x, mu, UT_chol) ) ;
// } // This function is checked with "dmvnorm" in mvtnorm package on 2018/01/26
// 
// arma::vec classIO::test_rMVN_fn(arma::vec mu, arma::mat sigma_mat){
//   return( Main.rMVN_fn(mu, sigma_mat) ) ; 
// } // This function is checked on 2018/01/27
// arma::vec classIO::test_rMVN_UT_chol_fn(arma::vec mu, arma::mat UT_chol){
//   return( Main.rMVN_UT_chol_fn(mu, UT_chol) ) ; 
// } // This function is checked on 2018/01/27
// 
// arma::mat classIO::test_rIW_fn(int nu_, arma::mat Phi_){
//   return( Main.rIW_fn( nu_, Phi_ ) ) ; 
// } // This function is checked on 2018/01/27
//   
// int classIO::test_rdiscrete_fn(arma::vec Prob){ 
//   return( Main.rdiscrete_fn(Prob) );
// } // This function is checked on 2018/01/26
