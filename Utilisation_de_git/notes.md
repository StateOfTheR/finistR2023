
Lien des précédents bootcamp : https://stateofther.netlify.app/#bootcamp

# Ateliers pour moi

- analyse multivariée
- git
- parallélisation Rcpp
- tidymodel
- dataviz
- nouveauté quarto
- analyse multi-tableaux 


gestion de la bibliographie : a priori il y a déjà un article dessus avec un précédent bootcamp R

quand on créer une nouvelle branche à partir d'une branche X, il copie tout ce qui est dans la branche X dans la nouvelle branche

Voir comment récupérer la mauvaisse branche qu'on a créer avec Caroline

# Fonctionnement de la semaine

L'idée de cette semaine est de travailler sur divers sujet et à chaque fois de faire un compte rendu qui fera ensuite parti d'un site web.
Le site sera créer à partir d'un dépôt github dont le lien est le suivant : 
https://github.com/StateOfTheR/finistR2023

Le site sera disponible à l'adresses suivante : https://stateofther.github.io/finistR2023/

Pour commencer cette semaine, j'ai fait parti d'un petit groupe dont le but était de revoir git, en particulier de manière à être capable de participer à la construction d site. On commencera donc par cloné le dépot, ensuite on fera une nouvelle branche "atelier_sur_git", dans laquelle on fera un Rmd pour notre compte rendu, que l'on soumettra via une pull request.

# Git

regarder nb.stripout pour gérer les notebook avec git (Correntin)

chercher le nom du logiciel pour git pour apprendre. 

Explication de l'arbre : 
  le premier "merge conflit" ne correspond à aucune branche divergente car il a été géré avec un rebase. Et rebase remet les choses en linéaires.
  les deux derniers "Merge branch ..." correspondent au git pull suivi d'une résolution de conflit.
  à ce moment la il fait un point dans l'arbre, comme si c'était un commit. 

on a fait git pull, ça n'a pas marché : fatal: Not possible to fast-forward, aborting.
  on ne sait pas très bien pourquoi

on a fait git pull --rebase
  on a eu des conflits : CONFLICT (content): Merge conflict in Utilisation_de_git/Utilisation_de_git.Rmd
  Pour les résoudre on a ouvert le fichier Rmd, on a supprimer les "=" et les ">", on a sauvegarder, et on a eu un git status u n peu bizare : cf screenshot 
  on a fait add et commit, ça a marché, puis on a essayé de push, on a eu l'erreur suivante : 
  cf screenshot (on n'était sur aucune branche car un rebase était en cours)

On a fait git rebase --continue, ça a terminé le rebase, et ensuite on a fait un push qui a bien marché

En fait on aurait du faire git config pull.rebase false pour qu'il configure cette méthode de gestion des conflits au moment de pull.

On va maintenant refaire un test avec cette nouvelle configuration

gitk pour voir les branches 

quand on fait git branch, il affiche juste les branches sur lesquelles on s'est déjà placé. quand on fait git branch -a, a pour all j'imagine, il les affiche toutes, avec des couleurs pour les branches sur lesquelles on n'a jamais été (remote)


pour mon souci perso, il faut que j'essaye de supprimer les derniers commit en particulier

# Journal

Lundi après-midi on a travaillé sur Git.

# FIN
