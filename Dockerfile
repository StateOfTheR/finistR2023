## Source docker
FROM rocker/geospatial:4

### JULIA

## Copie Julia's tar.gz et le decompresse
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.0-linux-x86_64.tar.gz && \
    tar -xvzf julia-1.7.0-linux-x86_64.tar.gz && \
    ## sauvegarde dans les programmes
    cp -r julia-1.7.0 /opt/ && \
    ## active julia (lien symbolique)
    ln -s /opt/julia-1.7.0/bin/julia /usr/local/bin/julia
RUN julia julia_packages.jl

## import de pandoc
RUN export DEBIAN_FRONTEND=noninteractive; apt-get -y update \
    && apt-get install -y pandoc \
    pandoc-citeproc

### packages R

## Installation dependances R
ENV R_CRAN_WEB="https://cran.rstudio.com/"
RUN R -e "install.packages('INLA',repos=c(getOption('repos'),INLA='https://inla.r-inla-download.org/R/stable'), dep=TRUE)"
RUN R -e "install.packages(c('remotes','microbenchmark','purrr','BiocManager','httr','cowplot','torch','PLNmodels','torchvision','reticulate','inlabru', 'lme4', 'ggpolypath', 'RColorBrewer', 'geoR','tidymodels', 'brulee', 'reprex','poissonreg','ggbeeswarm', 'tictoc', 'bench', 'circlize', 'JuliaCall', 'GeoModels'))"
RUN R -e "BiocManager::install('BiocPkgTools')"
RUN R -e "torch::install_torch(type = 'cpu')"
RUN R -e "install_julia()"

### librairies Ubuntu pour les projets python ?

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
  jags \
  mercurial gdal-bin libgdal-dev gsl-bin libgsl-dev \
  libc6-i386

### Jupyter Python torch jax etc

RUN apt-get install -y --no-install-recommends unzip python3-pip dvipng pandoc wget git make python3-venv && \
    pip3 install jupyter jupyter-cache flatlatex matplotlib && \
    apt-get --purge -y remove texlive.\*-doc$ && \
    apt-get clean
## Dependance python

RUN pip3 install jax jaxlib torch numpy matplotlib pandas scikit-learn torchvision torchaudio pyplnmodels optax

