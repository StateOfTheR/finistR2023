// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// Logistic loss
// [[Rcpp::export]]
// const : pas de changement de valeur dans la fonction
// arma:: : se sont les classes de Armadillo
// using namespace arma to erase all arma::
// & :  appel sans copie donc plus rapide
double loss_cpp( const arma::vec theta, const arma::vec& y, const arma::mat& x ) {
  arma::vec odds(x.n_rows);
  odds = x * theta;
  double log_lik;
  log_lik = arma::dot(y, odds) - arma::sum(arma::log(1 + arma::exp(odds)));
  return(-log_lik);
}
