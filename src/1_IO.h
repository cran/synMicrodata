#if !defined(_IO_H)
#define _IO_H

#include <RcppArmadillo.h>
#include "2_DataMain.h"

class classIO {
	
 public:
   
  classIO(arma::vec max_R_S_K_) ;
	~classIO() ; //destructor
	void Initialization() ; 
	 
  arma::mat GetY_mat() ;   void SetY_mat(arma::mat Y_mat_) ;  
	arma::mat GetX_mat() ;   void SetX_mat(arma::mat X_mat_) ;  
	arma::mat GetY_NA_mat() ;   void SetY_NA_mat(arma::mat Y_NA_mat_) ;  
	arma::mat GetX_NA_mat() ;   void SetX_NA_mat(arma::mat X_NA_mat_) ;  
	arma::vec GetD_l_vec() ;   void SetD_l_vec(arma::vec D_l_vec) ;  
  int Getmsg_level() ;  void Setmsg_level(int msg_level_) ;
  std::string Getwhere_we_are() ; void Setwhere_we_are(std::string where_we_are_) ; 
  
  void Iterate() ; void Run(int n_iter_) ; 
  
  // NEWLY ADDED (begin) //
  void Synthesis() ; 
  arma::mat GetSynt_Y_mat() ;
  arma::mat GetSynt_X_mat() ; 
  // NEWLY ADDED (end) // 
  
  
  // To check the code 
  
  arma::cube GetBeta_cube() ; void SetBeta_cube(arma::cube Beta_cube_) ;
  arma::vec Getr_i_vec() ; void Setr_i_vec(arma::vec r_i_vec_) ;
  arma::vec Gets_i_vec() ; void Sets_i_vec(arma::vec s_i_vec_) ;
  arma::vec Getk_i_vec() ; void Setk_i_vec(arma::vec k_i_vec_) ;
  arma::cube Getpsi_cube() ; void Setpsi_cube(arma::cube psi_cube_) ;
  
  arma::mat Gettest_Y_std_synt() ; void Settest_Y_std_synt(arma::mat test_Y_std_synt_) ; 
  
 private:
   
   classData Data ; classMain Main ;
   
   int IterCount ; 
   arma::vec RandVec ; 

};

#endif  //_classIO_H

