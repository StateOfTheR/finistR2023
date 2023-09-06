## Source docker
FROM rocker/geospatial:4

### JULIA

## Copy Julia's tar.gz and install it
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz && \
    tar zxvf julia-1.9.3-linux-x86_64.tar.gz && \
    ## Connect to Julia's directory link with real jula's bin
    cp -r julia-1.9.3 /opt/ && \
    ln -s /opt/julia-1.9.3/bin/julia /usr/local/bin/julia

## non Interactive terminal for this docker for the site
RUN export DEBIAN_FRONTEND=noninteractive; apt-get -y update \
    && apt-get install -y pandoc \
    pandoc-citeproc

### R packages

## Defining web acces for CRAN
ENV R_CRAN_WEB="https://cran.rstudio.com/"
RUN R -e "install.packages('INLA',repos=c(getOption('repos'),INLA='https://inla.r-inla-download.org/R/stable'), dep=TRUE)"
RUN R -e "install.packages(c('remotes','microbenchmark','purrr','BiocManager','httr','cowplot','torch','PLNmodels','torchvision','reticulate','inlabru', 'lme4', 'ggpolypath', 'RColorBrewer', 'geoR','tidymodels', 'brulee', 'reprex','poissonreg','ggbeeswarm', 'tictoc', 'bench', 'circlize', 'JuliaCall', 'GeoModels'))"
RUN R -e "BiocManager::install('BiocPkgTools')"
RUN R -e "torch::install_torch(type = 'cpu')"
RUN R -e "JuliaCall::install_julia()"

### Ubuntu libraries (for python ?)

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
  jags \
  mercurial gdal-bin libgdal-dev gsl-bin libgsl-dev \
  libc6-i386

### Jupyter Python torch jax etc

## Downloading Python
RUN apt-get install -y --no-install-recommends unzip python3-pip dvipng pandoc wget git make python3-venv && \
    pip3 install jupyter jupyter-cache flatlatex matplotlib && \
    apt-get --purge -y remove texlive.\*-doc$ && \
    apt-get clean

RUN pip3 install jax jaxlib torch numpy matplotlib pandas scikit-learn torchvision torchaudio pyplnmodels optax

## Install Julias package (IJulia <-- to connect with jupyter)
RUN julia -e 'import Pkg; Pkg.add("GeoDataFrames"); Pkg.add("Distributed"); Pkg.add("StatsAPI"); Pkg.add("Plots"); Pkg.add("DelimitedFiles"); Pkg.add("DataFrames"); Pkg.add("CSV"); Pkg.add("GeoStats"); Pkg.add("Rasters"); Pkg.add("Shapefile"); Pkg.add("GeoTables"); Pkg.add("CairoMakie"); Pkg.add("IJulia")'

## Check if jupyter find julia to then make it work in quarto !
# RUN quarto check jupyter
# RUN julia -e 'using GeoDataFrames'
