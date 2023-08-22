---
title: "FinistR : Utilisation de git"
date: "du 21 au 25 août 2023"
output: 
  html_document:
   toc: true
   toc_float: true
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, cache = TRUE)
library(tidyverse)
```

# Introduction

L'objectif de cet atelier est de progresser dans notre utilisation de git. En particulier pour ce bootcamp, on veut voir comment intéragir avec le dépot finisR2023 (pull request, proposer un fichier markdown, ...) 


# Historique

Dans la partie qui suit, on a travaillé avec deux ordinateurs et on a eu des conflits. On va voir comment on les a géré.

On commence par cloner le dépôt StateOfTheR/finistR2023 :
   
``git clone git@github.com:StateOfTheR/finistR2023.git``

On créer une nouvelle branche pour cet atelier git : 

``git branch Utilisation_de_git``

On se déplace sur cette branche : 

``git checkout Utilisation_de_git``

On créer un fichier Rmd dans un nouveau dossier dédié à cet atelier : 

    mkdir Utilisation_de_git
    cd Utilisation_de_git 
    nvim Utilisation_de_git.Rmd



# Brainstorming

- faire un repo propre, avec des logos, des sections, ...


# Commandes à suivre

- Créer une nouvelle branche : git branch nom_branche
- Se déplacer sur la nouvelle branche : git checkout nom_branche
- Ajouter le fichier RMD : git add
- Pusher la nouvelle branche : git push --set-upstream origin nom_branche



# Test conflit

test


# Test conflits encore et encore

Ce coup ci c'est la bonne

#Test conflit v2 
pull, push 



# Test Annaig conflit 
test 

# Annaig fait des conflits 

