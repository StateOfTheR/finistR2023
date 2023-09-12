## Source docker
FROM rocker/geospatial:4

### JULIA


## Copie Julia's tar.gz et le decompresse
## 1.8.3 marche bien avec quarto en local sinon ne fonctionne pas ...
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.3-linux-x86_64.tar.gz && \
    tar zxvf julia-1.8.3-linux-x86_64.tar.gz && \
    ## sauvegarde dans les programmes
    cp -r julia-1.8.3 /opt/ && \
    ln -s /opt/julia-1.8.3/bin/julia /usr/local/bin/julia

## Installe les depenances Julias  (IJulia <- connection avec jupyter) Indispensable pour quarto
RUN julia -e 'import Pkg; Pkg.add("GeoDataFrames"); Pkg.add("Distributed"); Pkg.add("StatsAPI"); Pkg.add("Plots"); Pkg.add("DelimitedFiles"); Pkg.add("DataFrames"); Pkg.add("CSV"); Pkg.add("GeoStats"); Pkg.add("Rasters"); Pkg.add("Shapefile"); Pkg.add("GeoTables"); Pkg.add("CairoMakie"); Pkg.add("WGLMakie"); Pkg.add("IJulia")'


## import de pandoc
RUN export DEBIAN_FRONTEND=noninteractive; apt-get -y update \
    && apt-get install -y pandoc \
    pandoc-citeproc

### packages R

## Installation dependances R
ENV R_CRAN_WEB="https://cran.rstudio.com/"
RUN R -e "install.packages('INLA',repos=c(getOption('repos'),INLA='https://inla.r-inla-download.org/R/stable'), dep=TRUE)"
RUN R -e "install.packages(c('dyplr','ggplot2','remotes','microbenchmark','purrr','BiocManager','httr','cowplot','torch','PLNmodels','torchvision','reticulate','inlabru', 'lme4', 'ggpolypath', 'RColorBrewer', 'geoR','tidymodels', 'brulee', 'reprex','poissonreg','ggbeeswarm', 'tictoc', 'bench', 'circlize', 'JuliaCall', 'GeoModels','sp','terra','gstat','sf'))"
RUN R -e "BiocManager::install('BiocPkgTools')"
RUN R -e "torch::install_torch(type = 'cpu')"
RUN R -e "JuliaCall::install_julia()"
## Installing from R JuliaCall package
# RUN R -e "invisible(purrr::map(c('GeoDataFrames','Distributed', 'StatsAPI', 'Plots', 'DelimitedFiles', 'DataFrames', 'CSV', 'GeoStats', 'Rasters',  'Shapefile','GeoTables', 'CairoMakie','WGLMakie'),JuliaCall::julia_install_package))"

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

## Check if jupyter find julia to then make it work in quarto !
# RUN quarto check jupyter
# RUN julia -e 'using GeoDataFrames'
