
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
  
  
RUN R -e "install.packages('INLA',repos=c(getOption('repos'),INLA='https://inla.r-inla-download.org/R/stable'), dep=TRUE)"
RUN R -e "install.packages('reticulate')"
RUN R -e "install.packages(c('inlabru', 'lme4', 'ggpolypath', 'RColorBrewer', 'geoR'))"
RUN R -e "install.packages(c('tidymodels', 'brulee', 'reprex'))"
RUN R -e "install.packages(c('poissonreg'))"
RUN apt-get install -y --no-install-recommends unzip python3-pip dvipng pandoc wget git make python3-venv && \
    pip3 install jupyter jupyter-cache flatlatex matplotlib && \
    apt-get --purge -y remove texlive.\*-doc$ && \
    apt-get clean

RUN pip3 install jax jaxlib optax torch numpy matplotlib pandas scikit-learn torchvision torchaudio 
RUN pip3 install pyplnmodels

    