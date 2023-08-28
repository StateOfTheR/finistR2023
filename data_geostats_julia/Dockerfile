
FROM rocker/geospatial:4
RUN export DEBIAN_FRONTEND=noninteractive; apt-get -y update \
 && apt-get install -y pandoc \
    pandoc-citeproc
RUN R -e "install.packages('remotes')"
RUN R -e "install.packages('microbenchmark')"
RUN R -e "install.packages('purrr')" # map function
RUN R -e "install.packages('BiocManager')" # map function
RUN R -e "BiocManager::install('BiocPkgTools')" 
RUN R -e "install.packages('httr')" # GET function
ENV R_CRAN_WEB="https://cran.rstudio.com/" 
RUN R -e "install.packages('cowplot')" # GET function
RUN R -e "install.packages('torch')"
RUN R -e "torch::install_torch(type = 'cpu')"
RUN R -e "install.packages('PLNmodels')"
RUN R -e "install.packages('torchvision')"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
  jags \
  mercurial gdal-bin libgdal-dev gsl-bin libgsl-dev \ 
  libc6-i386
  
RUN R -e "install.packages('rjags')"
RUN R -e "install.packages('nimble')"
RUN R -e "install.packages('ggmcmc')"
RUN R -e "install.packages('rstan')"
RUN R -e "install.packages('coda')"
RUN R -e "install.packages('reticulate')"



