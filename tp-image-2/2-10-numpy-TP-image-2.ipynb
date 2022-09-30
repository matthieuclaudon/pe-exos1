# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-hidden,-heading_collapsed,-run_control,-trusted
#     cell_metadata_json: true
#     notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version,
#       -jupytext.text_representation.format_version, -language_info.version, -language_info.codemirror_mode.version,
#       -language_info.codemirror_mode, -language_info.file_extension, -language_info.mimetype,
#       -toc
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#   nbhosting:
#     title: suite du TP simple avec des images
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %%
from IPython.display import HTML
HTML(url="https://raw.githubusercontent.com/ue12-p22/python-numerique/main/notebooks/_static/style.html")



# %% [markdown]
# # suite du TP simple avec des images
#
# merci à Wikipedia et à stackoverflow
#
# **le but de ce TP n'est pas d'apprendre le traitement d'image  
# on se sert d'images pour égayer des exercices avec `numpy`  
# (et parce que quand on se trompe ça se voit)**

# %%
import numpy as np
from matplotlib import pyplot as plt

# %% [markdown] {"tags": ["framed_cell"]}
# **notions intervenant dans ce TP**
#
# sur les tableaux `numpy.ndarray`
#
# * `reshape()`, tests, masques booléens, *ufunc*, agrégation, opérations linéaires sur les `numpy.ndarray`
# * les autres notions utilisées sont rappelées (très succinctement)
#
# pour la lecture, l'écriture et l'affichage d'images
#
# * utilisez `plt.imread`, `plt.imshow`
# * utilisez `plt.show()` entre deux `plt.imshow()` dans la même cellule
#
# **note**
#
# * nous utilisons les fonctions de base sur les images de `pyplot` par souci de simplicité
# * nous ne signifions pas là du tout que ce sont les meilleures  
# par exemple `matplotlib.pyplot.imsave` ne vous permet pas de donner la qualité de la compression  
# alors que la fonction `save` de `PIL` le permet
#
# * vous êtes libres d'utiliser une autre librairie comme `opencv`  
#   si vous la connaissez assez pour vous débrouiller (et l'installer), les images ne sont qu'un prétexte
#
# **n'oubliez pas d'utiliser le help en cas de problème.**

# %% [markdown]
# ## Création d'un patchwork

# %% [markdown]
# 1. Le fichier `rgb-codes.txt` contient une table de couleurs:
# ```
# AliceBlue 240 248 255
# AntiqueWhite 250 235 215
# Aqua 0 255 255
# .../...
# YellowGreen 154 205 50
# ```
# Le nom de la couleur est suivi des 3 valeurs de ses codes `R`, `G` et `B`  
# Lisez cette table en `Python` et rangez-la dans la structure qui vous semble adéquate.
# <br>
#
# 1. Affichez, à partir de votre structure, les valeurs rgb entières des couleurs suivantes  
# `'Red'`, `'Lime'`, `'Blue'`
# <br>
#
# 1. Faites une fonction `patchwork` qui  
#
#    * prend une liste de couleurs et la structure donnant le code des couleurs RGB
#    * et retourne un tableau `numpy` avec un patchwork de ces couleurs  
#    * (pas trop petits les patchs - on doit voir clairement les taches de couleurs  
#    si besoin de compléter l'image mettez du blanc  
#    (`numpy.indices` peut être utilisé)
# <br>
# <br>   
# 1. Tirez aléatoirement une liste de couleurs et appliquez votre fonction à ces couleurs.
# <br>
#
# 1. Sélectionnez toutes les couleurs à base de blanc et affichez leur patchwork  
# même chose pour des jaunes  
# <br>
#
# 1. Appliquez la fonction à toutes les couleurs du fichier  
# et sauver ce patchwork dans le fichier `patchwork.jpg` avec `plt.imsave`
# <br>
#
# 1. Relisez et affichez votre fichier  
#    attention si votre image vous semble floue c'est juste que l'affichage grossit vos pixels
#    
# vous devriez obtenir quelque chose comme ceci
# <img src="patchwork-all.jpg" width="200px">

# %% [markdown]
# ### Question 1 : 

# %%
file = open("rgb-codes.txt", 'r')

# %%
for line in file:
    print(line)
file.close()

# %% [markdown]
# On range tout cela dans un dictionnaire indexé par les noms des couleurs : 

# %%
file = open("rgb-codes.txt", 'r')
colors = {}
for line in file:
    L = line.split()
    colors[L[0]] = [int(L[i]) for i in range(1, 4)]
file.close()

# %% {"scrolled": true}
colors

# %% [markdown]
# ### Question 2 :

# %% {"scrolled": true}
colors['Red'], colors['Lime'], colors['Blue']


# %% [markdown]
# ### Question 3 :

# %% [markdown]
# *NB :* **l'affichage sur plusieurs pixels** de chaque carré de couleur sera traité à la **Q6** où `patchwork`sera modifié. 

# %%
def trouve_dim(n): # renvoie (nb_lgn, nb_col) d'un rectangle qui contient n ou plus éléments
    nb_col = 0
    while True:
        if nb_col**2 == n:
            return (nb_col, nb_col) # carré parfait
        elif nb_col**2 > n:
            if (nb_col-1)*nb_col >= n:
                return (nb_col-1, nb_col) # rectangle
            else:
                return (nb_col, nb_col)
        nb_col += 1


def patchwork(L): # L est une liste de couleurs. 
    n = len(L)
    nl, nc = trouve_dim(n)
    pixels_en_liste = [colors[elem] for elem in L] # liste brute des pixels
    for _ in range(nl*nc-n): # on prépare les cases blanches. 
        pixels_en_liste.append([255]*3)     
    pixels = [[pixels_en_liste[nc*i+j] for j in range(nc)] for i in range(nl)] # organisé pour préparer le tableau
    tab = np.array(pixels, dtype='uint8')
    return plt.imshow(tab)


# %%
trouve_dim(100)

# %%
trouve_dim(70)

# %% [markdown]
# ### Question 4 :

# %% [markdown]
# On crée déjà un dictionnaire `index`de la forme : numéro -> nom de la couleur. 

# %%
index = {}
color_names = list(colors.keys())
nb_colors = len(color_names)
for i in range(nb_colors):
    index[i] = color_names[i]


# %% [markdown]
# On pioche maintenant au hasard dans `index`pour constituer une liste de couleurs : 

# %%
def make_patchwork(n): # retourne un patchwork de n cases colorées
    L = [index[np.random.randint(0, nb_colors)] for _ in range(n)]
    patchwork(L)


# %% {"scrolled": false}
make_patchwork(40)

# %% [markdown]
# **Les pixels blancs sont bien rajoutés à la fin.**

# %%
make_patchwork(100)

# %% [markdown]
# ### Question 5 : 

# %% [markdown]
# On récupère les noms de toutes les couleurs (fonction déjà écrite) :

# %%
color_names

# %% [markdown]
# * **Couleurs blanches** : On ne converse, dans `white_colors`, que celles qui contiennent **White** : 

# %%
white_colors = []
for name in color_names:
    if 'White' in name:
        white_colors.append(name)

# %%
patchwork(white_colors);

# %% [markdown]
# * **Couleurs jaunes** : On ne converse, dans `yellow_colors`, que celles qui contiennent **Yellow** : 

# %%
yellow_colors = []
for name in color_names:
    if 'Yellow' in name:
        yellow_colors.append(name)

# %%
patchwork(yellow_colors);


# %% [markdown]
# ### Question 6 :
# On modifie légèrement le programme patchwork pour qu'il renvoie les valeurs brutes des pixels : 

# %%
def patchwork(L, dil=40): # L est une liste de couleurs, dil facteur de dilatation pour l'affichage de l'image. 
    n = len(L)
    nl, nc = trouve_dim(n)
    pixels_en_liste = [colors[elem] for elem in L] # liste brute des pixels
    for _ in range(nl*nc-n): # on prépare les cases blanches. 
        pixels_en_liste.append([255]*3)     
    pixels = [[pixels_en_liste[nc*i+j] for j in range(nc)] for i in range(nl)] # organisé pour préparer le tableau
    tab = np.array(pixels, dtype='uint8')
    tab2 = np.empty((dil*nl, dil*nc, 3), dtype='uint8')
    for i in range(nl):
        for j in range(nc):    
            tab2[dil*i:dil*i+dil, dil*j:dil*j+dil, :] = np.ones((dil, dil, 3)) * tab[i, j, :]
    return tab2


# %%
img = patchwork(color_names)
img.shape

# %%
plt.imsave("patchwork.jpg", img)

# %% [markdown]
# ### Question 7 :
# On peut modifier le facteur de dilatation `dil` dans la commande : `img = patchwork(color_names)` pour obtenir plus ou moins de pixels. 

# %%
img2 = plt.imread("patchwork.jpg")
plt.imshow(img2);

# %% [markdown]
# ## Somme des valeurs RGB d'une image

# %% [markdown]
# 0. Lisez l'image `les-mines.jpg`
#
# 1. Créez un nouveau tableau `numpy.ndarray` en sommant **avec l'opérateur `+`** les valeurs RGB des pixels de votre image  
#
# 2. Affichez l'image (pas terrible), son maximum et son type
#
# 3. Créez un nouveau tableau `numpy.ndarray` en sommant **avec la fonction d'agrégation `np.sum`** les valeurs RGB des pixels de votre image
#
# 4. Affichez l'image, son maximum et son type
#
# 5. Pourquoi cette différence ? Utilisez le help `np.sum?`
#
# 6. Passez l'image en niveaux de gris de type entiers non-signés 8 bits  
# (de la manière que vous préférez)
#
# 7. Remplacez dans l'image en niveaux de gris,   
# les valeurs >= à 127 par 255 et celles inférieures par 0  
# Affichez l'image avec une carte des couleurs des niveaux de gris  
# vous pouvez utilisez la fonction `numpy.where`
#
# 8. avec la fonction `numpy.unique`  
# regardez les valeurs différentes que vous avez dans votre image en noir et blanc

# %% [markdown]
# ### Question 0 :

# %%
img = plt.imread("les-mines.jpg")
plt.imshow(img);


# %% [markdown]
# ### Question 1 :

# %%
tab = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]

# %% [markdown]
# ### Question 2 :

# %%
plt.imshow(tab);

# %%
np.max(tab), type(tab)

# %% [markdown]
# ### Question 3 :

# %%
tab2 = np.sum(img, axis=2)

# %% [markdown]
# ### Question 4 :

# %%
plt.imshow(tab2);

# %%
np.max(tab2), type(tab2)

# %% [markdown]
# ### Question 5 :

# %%
tab, tab.shape

# %%
tab2, tab2.shape

# %% [markdown]
# * Avec `+`, on obtient pour chaque case une valeur souvent supérieure à 255. Elle est ramenée modulo 255 car le type est `uint8`. Les couleurs sont donc **faussées**.
# * A l'inverse, `tab2` est un `uint64` donc la valeur de chaque pixel est comprise entre 0 et 255 x 3. 

# %% [markdown]
# ### Question 6 :

# %%
bw = np.mean(img, axis=2).astype(np.uint8)
plt.imshow(bw, cmap='gray');

# %% [markdown]
# ### Question 7 :

# %%
T = np.dot((bw >= 127), 255)
plt.imshow(T, cmap='gray');

# %% [markdown]
# ### Question 8 :

# %% [markdown]
# * En nuances de gris :

# %%
tabl = np.unique(bw)
tabl, tabl.shape

# %% [markdown]
# On a toutes les nuances possibles.

# %% [markdown]
# * Avec que du noir et du blanc :

# %%
np.unique(T)

# %% [markdown]
# C'est normal. Seules deux couleurs sont présentes. 

# %% [markdown]
# ## Image en sépia

# %% [markdown]
# Pour passer en sépia les valeurs R, G et B d'un pixel  
# (encodées ici sur un entier non-signé 8 bits)  
#
# 1. on transforme les valeurs $R$, $G$ et $B$ par la transformation  
# $0.393\, R + 0.769\, G + 0.189\, B$  
# $0.349\, R + 0.686\, G + 0.168\, B$  
# $0.272\, R + 0.534\, G + 0.131\, B$  
# (attention les calculs doivent se faire en flottants pas en uint8  
# pour ne pas avoir, par exemple, 256 devenant 0)  
# 1. puis on seuille les valeurs qui sont plus grandes que `255` à `255`
# 1. naturellement l'image doit être ensuite remise dans un format correct  
# (uint8 ou float entre 0 et 1)

# %% [markdown]
# **Exercice**
#
# 1. Faites une fonction qui prend en argument une image RGB et rend une image RGB sépia  
# la fonction `numpy.dot` doit être utilisée (si besoin, voir l'exemple ci-dessous) 
#
# 1. Passez votre patchwork de couleurs en sépia  
# Lisez le fichier `patchwork-all.jpg` si vous n'avez pas de fichier perso
# 2. Passez l'image `les-mines.jpg` en sépia   

# %% {"scrolled": true}
# INDICE:

# exemple de produit de matrices avec `numpy.dot`
# le help(np.dot) dit: dot(A, B)[i,j,k,m] = sum(A[i,j,:] * B[k,:,m])

i, j, k, m, n = 2, 3, 4, 5, 6
A = np.arange(i*j*k).reshape(i, j, k)
B = np.arange(m*k*n).reshape(m, k, n)

C = A.dot(B)
# or C = np.dot(A, B)

A.shape, B.shape, C.shape

# %% [markdown]
# ### Question 1 :

# %%
corr = np.array([0.393, 0.349, 0.272, 0.769, 0.686, 0.534, 0.189, 0.168, 0.131]).reshape(3, 3)
def sepia(tab):
    tab2 = np.divide(tab.copy().astype(float), 255)
    sepia = tab2[:, :].dot(corr)
    sepia[sepia > 1] = 1
    plt.imshow(sepia)


# %% [markdown]
# ### Question 2 :

# %%
sepia(plt.imread("patchwork.jpg"))

# %% [markdown]
# ### Question 3 :

# %%
sepia(plt.imread("les-mines.jpg"))

# %% [markdown]
# ## Exemple de qualité de compression

# %% [markdown]
# 1. Importez la librairie `Image`de `PIL` (pillow)   
# (vous devez peut être installer PIL dans votre environnement)
# 1. Quelle est la taille du fichier 'les-mines.jpg' sur disque ?
# 1. Lisez le fichier 'les-mines.jpg' avec `Image.open` et avec `plt.imread`  
#
# 3. Vérifiez que les valeurs contenues dans les deux objets sont proches
#
# 4. Sauvez (toujours avec de nouveaux noms de fichiers)  
# l'image lue par `imread` avec `plt.imsave`  
# l'image lue par `Image.open` avec `save` et une `quality=100`  
# (`save` s'applique à l'objet créé par `Image.open`)
#
# 5. Quelles sont les tailles de ces deux fichiers sur votre disque ?  
# Que constatez-vous ?
#
# 6. Relisez les deux fichiers créés et affichez avec `plt.imshow` leur différence  

# %% [markdown]
# ### Question 1 :

# %%
from PIL import Image
import os

# %% [markdown]
# ### Question 2 :

# %% {"scrolled": true}
os.path.getsize("les-mines.jpg")

# %% [markdown]
# ### Question 3 :

# %%
img = plt.imread("les-mines.jpg")

# %%
img2 = Image.open("les-mines.jpg")

# %% [markdown]
# ### Question 4 :

# %%
plt.imshow(img);

# %%
img2

# %% [markdown]
# Les deux versions sont *a priori* proches. 

# %% [markdown]
# ### Question 5 :

# %%
plt.imsave("img_avec_imread.jpg", img)
img2.save("img_avec_Image.jpg", quality=100)

# %% [markdown]
# ### Question 6 :

# %%
os.path.getsize("img_avec_imread.jpg"), os.path.getsize("img_avec_Image.jpg")

# %% [markdown]
# L'image obtenue avec `Image` prend cinq fois plus de place. 

# %% [markdown]
# ### Question 7 :

# %%
img_a = plt.imread("img_avec_imread.jpg")
img_b = plt.imread("img_avec_Image.jpg")
diff = img_a - img_b
plt.imshow(diff);
