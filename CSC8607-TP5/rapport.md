# TP5 - Image Captioning avec ResNet50, LSTM et Attention

## Exercice 1 - Chargement des données et embeddings

### 1.1. Chargement du dataset Flickr30k

Les images sont prétraitées avec les opérations suivantes :

- Redimensionnement à `224 × 224` pixels.
- Conversion en tenseurs PyTorch.
- Normalisation à l'aide des moyennes et écarts-types d'ImageNet.


**Pourquoi redimensionner les images à 224 × 224 ?**

Le modèle ResNet50 utilisé dans la suite du TP est pré-entraîné sur ImageNet, avec un prétraitement standard utilisant des images de 224 × 224 pixels.

Le redimensionnement permet de fournir des images de dimensions homogènes, de construire des mini-batchs et de contrôler le coût de calcul ainsi que la mémoire GPU utilisée.

Il permet également d'obtenir une carte de caractéristiques spatiale de dimensions fixes, qui sera utilisée par le module d'attention.

En conservant les résolutions d'origine, les images pourraient avoir des dimensions différentes, ce qui compliquerait leur regroupement dans les mini-batchs et produirait des cartes spatiales de tailles variables.

ResNet50 peut techniquement accepter certaines autres résolutions, mais la taille fixe de 224 × 224 correspond au choix standard adapté à ce TP.


## Exercice 2 - Extraction de caractéristiques et Transfer Learning


### 2.1. Pourquoi geler les poids de ResNet50 ?

Nous avons gelé les paramètres du backbone avec :

```python
for param in cnn_features.parameters():
    param.requires_grad = False
```

Cette opération présente deux avantages principaux.

**Raison matérielle :**

Le gel des paramètres réduit le coût de la rétropropagation, car il n'est plus nécessaire de calculer et de conserver les gradients des poids du backbone.

Cela permet d'économiser de la mémoire GPU et du temps de calcul, ce qui est particulièrement important pour un modèle combinant ResNet50, attention et LSTM.

**Raison algorithmique :**

ResNet50 dispose déjà de caractéristiques visuelles générales acquises lors de son pré-entraînement sur ImageNet.

En conservant ces paramètres fixes, nous pouvons réutiliser ces représentations sans les modifier pendant l'entraînement du décodeur.

Cela réduit le risque de dégrader les représentations pré-entraînées et permet de concentrer l'apprentissage sur le module d'attention, le LSTM et la couche de prédiction des mots.

Cette approche est particulièrement adaptée lorsque les ressources de calcul sont limitées ou lorsque le nouveau dataset n'est pas suffisamment grand pour justifier le réentraînement complet du CNN.

### 2.2. Dimensions des caractéristiques extraites

Pour un batch de 32 images de dimensions `(3, 224, 224)`, le backbone de ResNet50 produit un tenseur de dimensions :

$$
\boxed{(32,\ 2048,\ 7,\ 7)}
$$

Les dimensions correspondent à :

- `32` : nombre d'images du batch.
- `2048` : nombre de canaux de caractéristiques visuelles.
- `7 × 7` : dimensions spatiales de la carte de caractéristiques.

Chaque image est ainsi représentée par 49 positions spatiales, chacune associée à un vecteur de 2048 caractéristiques.

Cette organisation est importante pour l'attention spatiale, car elle permet au modèle de calculer un poids pour chaque région de l'image.

La sortie du CNN ne constitue donc pas encore une description textuelle : elle fournit les informations visuelles nécessaires au futur décodeur LSTM.

## Exercice 3 - Module d'attention et LSTM personnalisé

### Interprétation du `context_vector` :

Le vecteur de contexte représente une combinaison pondérée des informations visuelles de l'image, donnant davantage d'importance aux régions sélectionnées par le mécanisme d'attention.

Dans un modèle Seq2Seq classique utilisant un unique vecteur global, les informations de l'image sont condensées dans une représentation fixe. Cette compression peut entraîner une perte d'informations spatiales et constitue un goulot d'étranglement.

L'attention vise à résoudre cette limitation en permettant au décodeur d'exploiter directement les caractéristiques des différentes régions plutôt que de dépendre exclusivement d'un vecteur global.


## Exercice 4 - Modèle global et boucle d'entraînement

### 4.1. Architecture globale du modèle

L'architecture comprend quatre parties principales :

1. **ResNet50 pré-entraîné :** extraction des caractéristiques visuelles de l'image sous la forme d'un tenseur de dimensions `(batch_size, 2048, 7, 7)`.
2. **Embeddings Word2Vec :** transformation des indices des mots en vecteurs pré-entraînés de dimension 300.
3. **LSTM avec attention spatiale :** combinaison des embeddings textuels, de l'état caché précédent et du contexte visuel.
4. **Couche linéaire de sortie :** projection des états cachés du LSTM vers les scores des mots du vocabulaire.

Pour une séquence d'entrée de 19 tokens et un état caché de dimension 512, la sortie du modèle possède les dimensions :

$$
(\text{batch\_size},\ 19,\ \text{vocab\_size})
$$

Chaque position temporelle fournit un score pour chaque mot possible.

### 4.2. Principe de l'entraînement et Teacher Forcing

L'entraînement utilise une prédiction auto-régressive du prochain mot, avec Teacher Forcing.

Les descriptions sont converties en séquences de longueur 20, puis séparées en deux tenseurs décalés :

```python
inputs = caps[:, :-1]
targets = caps[:, 1:]
```

Les 19 premiers tokens constituent les entrées du modèle et les 19 derniers les cibles.

À chaque position, le modèle prédit le mot suivant en exploitant les mots précédents et les caractéristiques visuelles.

Pendant l'entraînement, les vrais mots précédents sont utilisés comme entrées, indépendamment des prédictions du modèle.

Cette technique, appelée Teacher Forcing, facilite l'apprentissage des dépendances temporelles.

La fonction de perte utilisée est :

```python
criterion = nn.CrossEntropyLoss(ignore_index=0)
```

L'indice `0` représente le padding, ce qui permet de ne pas prendre en compte ces positions dans le calcul de la perte.

Nous utilisons l'optimiseur Adam avec un learning rate initial de 0.001.


Un Gradient Clipping supplémentaire avec une norme maximale de 1.0 est appliqué entre `backward()` et `step()` pour limiter les mises à jour excessives.



**Évolution des pertes**

| Époque | Train Loss | Validation Loss | Learning rate |
|---|---|---|---|
| 1 | 5,5697 | 4,8328 | 0.001 |
| 5 | 3,3019 | 4,3266 | 0.001 |
| 10 | 2,6187 | 4,3863 | 0.0001 |
| 15 | 2,5266 | 4,4043 | 0.00001 |

![Évolution des pertes d'entraînement et de validation](images/train_loss.png)
![Évolution des pertes d'entraînement et de validation](images/validation_loss.png)

**Analyse de la convergence**

Au cours des 15 époques, la Train Loss diminue continuellement, passant de 5,5697 à 2,5266. Cela indique que le modèle apprend progressivement à prédire les mots des descriptions présentes dans l'ensemble d'entraînement.

La Validation Loss diminue également pendant les premières époques, passant de 4,8328 à une valeur minimale de 4,3113 à l'époque 4.

Cependant, à partir de l'époque 5, elle commence à augmenter progressivement, pour atteindre 4,4043 à l'époque 15, alors que la Train Loss continue de diminuer.

Cet écart croissant entre les deux pertes constitue un signe de surapprentissage (overfitting) : le modèle améliore ses prédictions sur les exemples d'entraînement, mais cette amélioration ne se traduit plus par une meilleure généralisation aux images de validation.

Le planificateur StepLR réduit le learning rate de 0,001 à 0,0001 après l'époque 5, puis à 0,00001 après l'époque 10. La diminution du learning rate permet de réaliser des mises à jour plus petites, mais ne suffit pas ici à inverser l'augmentation progressive de la Validation Loss.

Le meilleur modèle selon la Validation Loss est donc celui de l'époque 4, avec une valeur de 4,3113.

L'utilisation d'un mécanisme d'Early Stopping aurait pu éviter de poursuivre l'entraînement au-delà des premières époques sans amélioration de la validation. La sauvegarde du meilleur checkpoint permet néanmoins de conserver les paramètres associés à la meilleure performance de validation.

Ces résultats montrent l'importance de surveiller conjointement les pertes d'entraînement et de validation : une diminution continue de la Train Loss ne garantit pas une amélioration de la capacité de généralisation du modèle.

### 4.5. Analyse des légendes générées aux époques 1 et 10

Afin d'étudier les progrès du modèle, nous observons les descriptions produites pendant l'entraînement.

Les prédictions avec Teacher Forcing sont calculées à partir des véritables mots précédents. Pour étudier la capacité du modèle à générer une légende de manière autonome, nous effectuons également une génération auto-régressive.

**Époque 1**

![Epoch1](images/epoch1.png)

**Époque 10**

![Epoch10](images/epoch10.png)


**Analyse :**

À l'époque 1, les légendes générées sont encore très rudimentaires.  
La prédiction avec Teacher Forcing contient de nombreuses répétitions et reste peu grammaticale, par exemple : *“little girl in in in in in the girl...”*.  
La génération auto-régressive est légèrement plus structurée, mais elle reste elle aussi répétitive et incomplète : *“little girl in blue shirt is holding her in the grass...”*.  
Le modèle semble donc avoir appris quelques mots fréquents du dataset, mais pas encore la structure correcte d'une légende.

À l'époque 10, une amélioration nette apparaît, surtout en génération auto-régressive.  
La sortie produite est plus longue, plus fluide et globalement plus cohérente : *“two young girls are sitting on the forest with trees in the woods with trees in the background...”*.  
Même si cette phrase contient encore des répétitions et certaines maladresses grammaticales, elle décrit une scène visuelle plausible et plus riche qu'à l'époque 1.

La prédiction avec Teacher Forcing à l'époque 10 reste imparfaite, avec des erreurs comme *“two girl”* ou *“fallen tongue out”*.  
Cela montre que la diminution de la loss ne garantit pas une qualité parfaite des légendes. Néanmoins, les résultats confirment que le modèle apprend progressivement à générer des descriptions plus cohérentes et plus informatives.

En résumé, entre l'époque 1 et l'époque 10, on observe :
- une meilleure structure grammaticale globale ;
- une plus grande diversité des mots ;
- une meilleure cohérence sémantique avec l'image ;
- mais encore des répétitions et des erreurs de formulation.

Ces observations montrent qu'une amélioration des métriques quantitatives s'accompagne ici d'un progrès qualitatif réel, surtout pour la génération auto-régressive.

### 4.6. Analyse du StepLR

Le planificateur de taux d'apprentissage utilisé est :

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)
```

Il multiplie le learning rate par 0.1 toutes les cinq époques.

Le planning appliqué est :

| Époques | Learning rate |
|---|---:|
| 1 à 5 | 0.001 |
| 6 à 10 | 0.0001 |
| 11 à 15 | 0.00001 |

![Évolution du learning rate](images/learning_rate.png)

**Que se passerait-il si nous conservions un learning rate constant de 0.001 ?**

Avec un learning rate constant, les mises à jour des paramètres conserveraient une amplitude globale relativement importante durant les dernières époques.

Cela pourrait provoquer des oscillations autour d'une région de faible perte ou rendre la convergence moins stable.

La diminution progressive du learning rate permet de réaliser des mises à jour plus petites et d'affiner les paramètres une fois que le modèle a appris les principales relations entre les images et les descriptions.

Cependant, un learning rate constant n'entraînerait pas nécessairement une divergence : son effet réel dépendrait des données, de l'optimiseur et de la dynamique des gradients.

**Pourquoi la décroissance du learning rate est-elle pertinente avec Adam ?**

Adam combine une estimation de la moyenne mobile des gradients et une estimation de leur second moment pour adapter les mises à jour des paramètres.

Malgré cette adaptation, le learning rate global demeure un facteur essentiel qui détermine l'échelle des mises à jour.

En fin d'entraînement, sa diminution permet d'effectuer des ajustements plus précis et peut réduire les oscillations résiduelles.

Cette stratégie est pertinente avec les optimiseurs utilisant le momentum ou des statistiques de gradients, comme Adam et RMSProp, car leurs estimations adaptatives ne remplacent pas totalement le contrôle du learning rate global.