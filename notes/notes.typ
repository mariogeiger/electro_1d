// Notes (in French) on why light "slows down" in matter, for a pulse: the coupled-sheet model, its exact
// solution, and the candidate figures of the article. Build the figures first (figures/), then, from the
// repository root: typst compile --root . notes/notes.typ out/notes.pdf

#set document(title: "Pourquoi la lumière « ralentit » : le cas d'une impulsion", author: ("Quentin Talon", "Mario Geiger"))
#set page(paper: "a4", margin: (x: 2.1cm, y: 2.2cm), numbering: "1 / 1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "fr")
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)", supplement: none)
#set heading(supplement: [section])
#set figure(numbering: none)
#show ref: it => if it.element != none and it.element.func() == math.equation [(#it)] else [#it]
#show figure.caption: set text(size: 9pt)
#show link: set text(fill: rgb("#1f5fa8"))

#let fig(path, cap, width: 100%) = figure(image("../out/figures/" + path, width: width), caption: cap)
#let card(body) = block(fill: rgb("#f3f6fa"), inset: 9pt, radius: 3pt, width: 100%, body)
#let Einc = $E_"inc"$
#let Emat = $E_"mat"$

#align(center)[
  #text(17pt, weight: "bold")[Pourquoi la lumière « ralentit » dans la matière :\ le cas d'une impulsion]
  #v(2pt)
  #text(11pt)[Notes d'exploration numérique et propositions d'illustrations\ pour un article (_Am. J. Phys._ ou _Eur. J. Phys._)]
  #v(4pt)
  #text(9.5pt, fill: luma(80))[Document de travail pour Quentin Talon et Mario Geiger --- rédigé par l'agent IA de Mario --- 28 octobre 2026]
]

#card[
  *En bref.*
  - Le code de Mario @rmario (#link("https://github.com/mariogeiger/electro_1d")[electro_1d]) tourne tel quel (GPU, ≈ 5 s par simulation). Je l'ai rejoué dans plusieurs variantes, et j'ai écrit un *solveur de référence indépendant*, exact et sans grille, qui calcule séparément l'onde incidente et le champ rayonné par chaque feuillet de charges.
  - *L'idée de Quentin est confirmée de façon quantitative.* À tout instant, le champ total est la somme exacte $E = Einc + Emat$. L'impulsion incidente continue à $c$ partout, y compris dans la matière. Le champ rayonné par la matière contient une *copie inversée* de l'impulsion incidente, qui se déplace à $c$ et l'annule (c'est le théorème d'extinction d'Ewald--Oseen), plus une impulsion *retardée*. Seule la somme semble aller à $c\/n_g$ (fig. A).
  - Le front de l'impulsion n'est jamais ralenti : rien n'existe au-delà du cône de lumière du front (fig. E). L'enveloppe, elle, est retardée de $(n_g-1)L\/c$ et la phase de $(n-1)L\/c$. Les deux retards mesurés concordent avec la théorie à 1 % et 0,02 % (fig. D et F).
  - *Point à corriger dans le code* : la grille colocalisée (différences centrées) émet une onde parasite à l'échelle de la maille, d'amplitude allant jusqu'à ≈ 50 % de celle de l'impulsion. Une grille décalée (Yee), qui ne change que deux lignes, ramène l'écart à ≈ 1 % (fig. H).
  - Je propose *9 illustrations* (S, A--H, section 6). Ma recommandation pour la figure clé : *A* (diagrammes espace-temps $E$, $Einc$, $Emat$), éventuellement couplée à *B* (instantanés).
]

= Le concept à démontrer <concept>

Les vidéos de 3Blue1Brown @r3b1b et de Looking Glass Universe @rlgu1 @rlgu2 reprennent l'argument de Feynman @rfeyn : chaque couche d'atomes, mise en oscillation par l'onde, rayonne une petite onde en quadrature. La somme « onde incidente + ondes rayonnées » est une onde déphasée, donc d'apparence plus lente. Cet argument est fait en *régime stationnaire* (onde monochromatique infinie). Il ne dit donc rien de ce qui arrive à une *impulsion* : qu'est-ce qui est retardé, l'enveloppe ou la phase ? Que devient le front ? Où est l'énergie pendant la traversée ?

L'idée de Quentin se formule exactement ainsi, parce que les équations de Maxwell sont linéaires :
$ E(x,t) = underbrace(Einc(x,t), "vide : à " c " partout") + underbrace(Emat(x,t), "rayonné par les charges : à " c " aussi") . $ <split>
Aucun terme ne se propage moins vite que $c$. Le ralentissement apparent est une propriété de la *somme*. La simulation permet de calculer et de *montrer séparément* chacun des deux termes, ce qu'aucune des explications citées ne fait pour une impulsion.

#card[
  *Formulation proposée pour l'article.* « L'impulsion originale n'est pas ralentie : elle traverse la matière à $c$, mais elle y est annulée par le champ que rayonnent les charges qu'elle met en mouvement. Ces mêmes charges rayonnent une impulsion retardée, et c'est elle qu'on observe. »
]

= Pourquoi une simulation 1D est légitime <why1d>

Le passage en 1D n'est pas seulement une économie de calcul : pour le problème posé, c'est une réduction *exacte*.

+ *Symétrie.* Une onde plane en incidence normale sur un milieu homogène dans le plan transverse $(y,z)$ produit des champs qui ne dépendent que de $(x,t)$. C'est la géométrie d'un faisceau large devant la longueur d'onde qui tombe sur une lame. La symétrie vaut à chaque instant, donc pour n'importe quelle forme temporelle, et en particulier pour une impulsion.
+ *Un plan d'atomes est un feuillet.* Feynman (vol. I, §30-7) somme les ondes sphériques des dipôles d'un plan de charges qui oscillent en phase. Il obtient une onde plane, $E = -K(t - |x|\/c)\/(2 epsilon_0 c)$, où $K$ est la densité surfacique de courant. C'est *exactement* la fonction de Green de Maxwell en 1D. Le modèle de Mario (des plans de charges reliés à des ressorts) est donc le modèle de Feynman, sans l'approximation « loin du plan ».
+ *Ce que le 1D perd.* La granularité dans le plan : la diffusion incohérente (Rayleigh, le bleu du ciel) et les corrections de champ local (Lorentz--Lorenz). Le 1D décrit le faisceau *cohérent*, celui qui définit l'indice. Il donne exactement la dispersion de Lorentz d'un milieu dilué, $n^2 = 1 + omega_p^2\/(omega_0^2 - omega^2)$. Les angles de réfraction (Snell--Descartes) sont hors de portée.
+ *Bonus du 1D.* La réaction de rayonnement d'un feuillet est une simple friction linéaire, $-q K\/2$ : elle est exacte et sans la pathologie d'Abraham--Lorentz. L'énergie se conserve exactement entre champ et oscillateurs.

= Le modèle et les équations utiles <model>

Unités : $c = epsilon_0 = 1$ (comme dans le code). Le feuillet $k$, à la position $x_k$, porte une charge $q$ et une masse $m$ par unité de surface, et il est rappelé par un ressort de pulsation $omega_0$. Son courant de surface vaut $K_k = q dot(y)_k$. Seule la combinaison $epsilon = q^2\/m$ intervient en régime linéaire. Une paire $plus.minus q$ du notebook se comporte comme un feuillet avec $epsilon = 2 q^2 \/ m$.

*Champ d'un feuillet et superposition exacte.*
$ E(x,t) = Einc(x - c t) - 1/2 sum_k K_k (t - |x - x_k|\/c), quad m dot.double(y)_j = q E(x_j, t) - m omega_0^2 y_j . $ <master>
Le système se referme sur les seules charges : chaque feuillet ressent l'onde incidente plus les champs *retardés* des autres. Le terme $k = j$ est la réaction de rayonnement. Aucune équation de Maxwell n'est à résoudre sur une grille. Ce sont des équations de Newton avec retards, que l'on peut résoudre en ≈ 30 lignes de code.

*Un seul feuillet : le retard de Feynman, mais en temps.* Si le spectre de l'impulsion est loin sous la résonance ($omega << omega_0$), on a $y approx q E\/(m omega_0^2)$, donc
$ E_"feuillet" = -tau partial_t E, quad tau = epsilon/(2 omega_0^2), quad E + E_"feuillet" approx E(t - tau). $ <tau>
Le feuillet rayonne la *dérivée temporelle* du champ qu'il ressent. Ajouter $-tau partial_t E$ revient à *décaler* l'impulsion de $tau$, quelle que soit sa forme. Vers l'arrière, le même champ $-tau partial_t E$ est l'écho réfléchi.

*Beaucoup de feuillets.* Pour un empilement périodique de pas $d$ (exact, Kronig--Penney avec des deltas) :
$ cos(k d) = cos(omega d) - sigma(omega) sin(omega d), quad sigma = (epsilon omega)/(2 (omega_0^2 - omega^2)) quad arrow.r_(d -> 0) quad n^2 = 1 + omega_p^2/(omega_0^2 - omega^2), quad omega_p^2 = epsilon/d . $
Après une épaisseur $L$, l'enveloppe est retardée de $(n_g - 1) L$ et la phase de $(n - 1) L$, avec $n_g = n + omega dif n \/ dif omega$. Pendant la traversée, la matière détient la fraction $1 - (1 + n^2)\/(2 n n_g)$ de l'énergie.

*Paramètres du notebook.* Ils donnent $epsilon = 2\/0.9$, $omega_0 = 5$, $d = 5\/29$, $omega_p = 3.59$, une porteuse $omega = 3$, et 30 paires ($L = 5.17$). Pour l'empilement exact : $n = 1.349$ et $n_g = 1.710$ (le continu de Lorentz donne 1.344 et 1.681). Bande interdite : $5 < omega < 6.16$.

= Implémentation et validation <impl>

*Code de Mario.* Il intègre en JAX (float64) les champs $E_y, E_z, B_y, B_z$ sur une grille de 4096 points, en conditions périodiques, avec des dérivées centrées. Les feuillets sont déposés sous forme de « chapeaux » sur deux mailles ; $E_x$ est donné analytiquement par Gauss. Le tout passe par `diffrax` (Tsit5, pas adaptatif) avec 30 paires $plus.minus 1$ de même masse. C'est une méthode « brute force » : rien de la physique de l'indice n'y est mis à la main, et c'est sa force pédagogique.

*Solveur de référence (nouveau).* L'équation @master est linéaire et invariante dans le temps. En fréquence, les courants vérifient le système $N times N$
$ (chi^(-1)(omega) delta_(j k) + 1/2 e^(i omega |x_j - x_k|)) K_k = Einc(x_j, omega), quad chi(omega) = (-i omega epsilon)/(omega_0^2 - omega^2) . $
Je le résous pour chaque fréquence, puis je reviens en temps par FFT. Le décalage complexe $omega + i eta$ supprime le repliement (erreur $tilde e^(-eta T)$). Il n'y a ni grille ni boîte, et $Einc$ et $Emat$ sont séparés par construction. Le solveur reproduit la transmission analytique d'un feuillet à $10^(-6)$ près. Le code (`sheets.py`) et tous les scripts sont dans mon répertoire de travail (voir l'annexe).

#fig("figH_validation.png", [*(H) Validation.* Gauche : $E_y$ à $t = 14$ dans le cas du notebook (amplitude $10^(-3)$, régime linéaire, remise à l'échelle). Grille colocalisée du notebook en rouge, grille décalée (Yee) en bleu, solution exacte en pointillés. Droite : écart maximal à la solution exacte au cours du temps. La grille colocalisée crée une onde parasite à l'échelle de la maille (≈ 50 %) ; la grille de Yee reste à ≈ 1 %. À amplitude 1 (celle du notebook), les charges atteignent $v approx 0.2 c$ et la force $v times B$ dévie la solution de ≈ 12 %.]) <figH>

#card[
  *Points à corriger dans le code (pour Mario).*
  + *Mode parasite.* Avec des dérivées centrées sur grille colocalisée, $omega(k) = sin(k Delta)\/Delta$ s'annule aussi en $k = pi\/Delta$. Il existe donc une branche non physique, de vitesse de groupe $-c$, que les courants très localisés des feuillets excitent fortement (ce sont les fines stries obliques de `output.png`). Le remède : $E$ aux nœuds et $B$ aux demi-nœuds, avec `dx` avant/arrière @ryee (variante testée : `electro_yee.py`).
  + *Tolérance.* `atol = 1e-6` est absolue : à petite amplitude, le bruit numérique domine. Il faut la mettre à l'échelle de l'amplitude.
  + *Linéarité.* L'amplitude 1 du notebook n'est pas « petite » ($v approx 0.2 c$, et le code est non relativiste). Pour l'article, mieux vaut travailler en régime linéaire, où le résultat est indépendant de l'amplitude.
  + *Bords périodiques.* L'écho réfléchi ressort de l'autre côté et croise l'impulsion transmise (coin en haut à droite de `output.png`). Il faut un domaine plus grand, des bords absorbants, ou le solveur exact.
]

= Ce que les simulations montrent <explore>

#fig("figA_spacetime_mario.png", [*(M) Cas du notebook* (solveur exact, même impulsion et mêmes 30 paires). Couleur : $E_y (x,t)$. Tirets : bords de la lame. Pointillés : trajectoire à $c$ du centre de l'impulsion. Trait plein : trajectoire prédite à $v_g = c\/n_g$ dans la lame. Le retard $(n_g - 1)L = 3.7$ est plus court que l'impulsion : dans $Emat$, la copie inversée et l'impulsion retardée se chevauchent encore.]) <figM>

Faits saillants, du plus important au plus subtil :
- *Décomposition.* Dans la lame et derrière elle, $Emat = -Einc("à" c) + E_"retardée"$. L'onde incidente n'est pas absorbée, elle est *annulée par superposition*. Cela se voit directement dès que la lame est assez épaisse pour séparer les deux impulsions (fig. A).
- *Accumulation.* Un feuillet de paires du notebook retarde de $tau approx 0.044$ seulement. Le retard croît linéairement avec le nombre $N$ de feuillets : 0,1238 par feuillet mesuré, contre 0,1224 prédit pour l'enveloppe ; 0,06024 contre 0,06023 pour la phase.
- *Phase et enveloppe sont retardées différemment* ($n = 1.35$, $n_g = 1.71$). L'explication « stationnaire » ne parle que de la phase.
- *Causalité.* Le champ est nul au-delà de $x = x_"front" + c t$ (à $10^(-5)$ près, limite de la discrétisation). Le front, et un petit précurseur de Sommerfeld, arrivent à $c$, et le gros du signal $(n_g - 1)L$ plus tard.
- *Régimes.* Sous la résonance, $n > 1$. Près de $omega_0$, $n_g$ atteint ≈ 16 et l'impulsion se déforme. Dans la bande interdite, il y a réflexion totale et onde évanescente. Au-dessus, $n < 1$ : la phase va *plus vite* que $c$, mais l'enveloppe reste plus lente, $n_g > 1$.
- *Énergie.* Pendant la traversée, 40 % de l'énergie est dans les oscillateurs (39 % prédit pour une onde quasi monochromatique). L'impulsion est comprimée d'un facteur $n_g$, et l'énergie totale est conservée à 0,3 % près.
- *Autres effets visibles.* Réflexion de Fresnel ($(n - 1)\/(n + 1) = 0.15$ en amplitude) à chaque interface, échos multiples dans la lame, et étalement avec dérive de fréquence (dispersion de vitesse de groupe) pour les lames épaisses.

= Illustrations candidates <choix>

Toutes les figures ci-dessous viennent du solveur exact. Les textes y sont en anglais pour pouvoir les reprendre telles quelles dans l'article ; couleurs, cadrages et paramètres sont faciles à changer. Sauf mention contraire, le milieu est celui du notebook, et la « lame épaisse » compte 60 paires au lieu de 30 ($L = 10.3$).

#fig("figS_schematic.png", [*(S) Schéma du modèle.* Feuillets chargés sur ressorts, impulsion incidente, champ rayonné par un feuillet, et superposition @master. Indispensable dans l'article, quelle que soit la figure clé.]) <figS>

#fig("figA_spacetime_thick.png", [*(A) Figure clé proposée : diagrammes espace-temps* du champ total $E$ (ce qu'on « voit »), de l'onde incidente seule $Einc$ (qui continue à $c$, comme sans matière) et du champ rayonné par la matière $Emat = E - Einc$. Lame épaisse, impulsion du notebook. Dans le panneau de droite, on lit l'idée de Quentin : une copie *inversée* de l'impulsion incidente (sur la ligne pointillée à $c$), qui l'éteint, plus l'impulsion retardée (trait plein, $v_g$ dans la lame).]) <figA>

#fig("figB_snapshots.png", [*(B) Instantanés* $Einc$ (tirets), $Emat$ (orange) et $E$ (noir) à quatre instants (lame épaisse grisée). À $t = 30$, devant l'impulsion transmise, $Einc$ et $Emat$ sont égaux et opposés : on y voit l'extinction à l'œil nu.]) <figB>

#fig("figC_single_sheet.png", [*(C) Le mécanisme, un feuillet à la fois.* Une impulsion gaussienne sans porteuse traverse un seul feuillet, volontairement fort ($tau = 0.15$). Le feuillet rayonne $approx -tau partial_t E$ (pointillés) ; ajouté à l'onde incidente, ce champ la décale de $tau$ (bleu pointillé). Le même champ, émis vers l'arrière, est l'écho (droite). C'est l'argument de Feynman, en temps et pour une forme quelconque. Écart résiduel : correction inertielle $omega^2\/omega_0^2$.]) <figC>

#fig("figD_layers.png", [*(D) Accumulation couche par couche.* Gauche : l'impulsion du notebook après $N = 0, 1, 2, dots, 64$ feuillets, en temps retardé (le front incident est en 0) ; l'enveloppe est en orange. Elle recule, s'étale et se met à dériver en fréquence, mais rien ne précède jamais le front. Droite : avec une impulsion à bande étroite, les retards d'enveloppe et de phase mesurés suivent $(n_g - 1) N d$ et $(n - 1) N d$.]) <figD>

#fig("figE_front.png", [*(E) Le front n'est jamais ralenti.* Gauche : $log_10 |E|$ (lame épaisse) ; le champ est nul (noir, $< 10^(-6)$) à droite du cône de lumière du front. Droite : sinusoïde allumée à $t = 0$ (problème de Sommerfeld), vue derrière la lame. Le précurseur arrive exactement à $c$, le signal principal $(n_g - 1)L$ plus tard (échelle logarithmique en bas).]) <figE>

#fig("figF_regimes.png", [*(F) Régimes de dispersion.* Haut : diagrammes espace-temps (30 paires) pour quatre porteuses : sous la résonance, près de la résonance, dans la bande interdite (miroir), et au-dessus ($n < 1$). Bas : $n(omega)$ et $n_g (omega)$ théoriques (empilement exact en trait plein, Lorentz continu en tirets) et valeurs *mesurées* sur les impulsions simulées (symboles). Le plateau de $n$ juste sous $omega_0$ est le bord de la zone de Brillouin ($k d = pi$) : un effet de la discrétisation en feuillets, absent du continu.]) <figF>

#fig("figG_energy.png", [*(G) Où est l'énergie ?* Gauche : énergie du champ (à gauche, dans et à droite de la lame) et énergie mécanique des oscillateurs. Droite : part de l'énergie de la lame détenue par la matière pendant la traversée, comparée à $1 - (1 + n^2)\/(2 n n_g)$.]) <figG>

#block(breakable: false)[#table(
  columns: (auto, 1fr, auto, auto),
  align: (center, left, left, center),
  stroke: 0.4pt + luma(170),
  inset: 5pt,
  table.header([*Fig.*], [*Message*], [*Public*], [*Mon avis*]),
  [S], [le modèle en un coup d'œil], [tous], [indispensable],
  [A], [l'onde incidente va à $c$ ; la matière l'annule et en émet une retardée], [tous], [*figure clé*],
  [B], [même message, en instantanés (plus lisible sur papier)], [tous], [complète A],
  [C], [un feuillet rayonne $-tau partial_t E$, d'où un retard], [licence], [très pédagogique],
  [D], [les retards s'additionnent ; phase et enveloppe diffèrent], [licence], [fort],
  [E], [causalité : le front va à $c$, précurseurs], [master], [fort],
  [F], [$n(omega)$, $n_g (omega)$ mesurés ; $v_phi > c$ inoffensif], [master], [fort],
  [G], [l'énergie est partagée avec la matière], [master], [bonus],
  [H], [validation numérique], [annexe], [annexe],
)]

*Recommandation.* La figure clé serait *A* : elle montre l'idée elle-même, sans formule. On peut y ajouter *B* en bas, pour qu'elle reste lisible sur papier, ou bien placer *B* juste après. Je suggère ensuite de construire l'argument avec *C → D → E → F*, en gardant *G* comme discussion et *H* en annexe. Une animation (`animation_pulse.gif`, jointe) pourrait servir de matériel supplémentaire.

= Proposition de plan d'article <plan>

+ *Introduction* : l'énigme du « ralentissement » ; Feynman et les vidéos ; ce qui manque (les impulsions) ; l'idée de Quentin @split.
+ *Modèle* : pourquoi le 1D est exact pour ce problème (@why1d) ; feuillets ; fonction de Green ; équation @master ; figure S.
+ *Un feuillet* : l'équation @tau ; figure C ; la réflexion comme même phénomène.
+ *Une lame* : la superposition en action, figure A (+ B) ; extinction d'Ewald--Oseen en temps ; figure D.
+ *Phase, groupe, front* : figures E et F ; le malentendu « absorption puis réémission » (la réponse est immédiate et cohérente, il n'y a pas d'« attente ») ; $v_phi > c$.
+ *Énergie* (G) et *discussion* : limites (3D, champ local, quantique), lien avec la lumière lente @rhau.
+ *Méthodes* : solveur de Maxwell sur grille et solveur par superposition, validation (H), code public.

*Format.* AJP (catégories _Papers_ ou _Notes and Discussions_, plus courte) et EJP publient tous deux ce type d'article pédagogique avec du matériel supplémentaire (code, animations). Longueurs et gabarits sont à vérifier dans les guides auteurs. Les deux attendent qu'on situe clairement l'apport par rapport à la littérature, d'où le point suivant.

*Littérature : à approfondir.* Ma recherche (rapide, sur arXiv seulement) montre que le « cas impulsion » n'est pas totalement vierge. Benedek _et al._ @rbenedek appliquent le théorème d'extinction à des impulsions ultracourtes. Mansuripur @rmansu démontre ce théorème avec des feuillets de dipôles, mais en régime harmonique. Grzesik @rgrzesik traite la lame comme un phénomène de rayonnement auto-cohérent. Les précurseurs de Sommerfeld--Brillouin @rsommerfeld @rbrillouin @rbrillouinbook @roughstun et Jackson §7.11 @rjackson traitent le front. Il faut aussi citer les classiques d'AJP sur Ewald--Oseen @rfearn @rballen. L'apport le plus défendable me paraît être *une démonstration microscopique, en temps et par simulation, qui sépare explicitement $Einc$ et $Emat$ pour une impulsion*, avec le front, la phase, le groupe et l'énergie dans un seul cadre exact et minimal.

= Décisions à prendre <questions>

+ Quelle figure clé (A, A+B, ou une autre) ? Quelles figures garder ?
+ Paramètres : garder exactement ceux du notebook (familiers), ou une lame plus épaisse ou plus dense pour séparer nettement les deux impulsions (A, B) ?
+ Méthode principale de l'article : le solveur de Maxwell sur grille (corrigé en Yee), le solveur par superposition @master (qui *est* l'idée), ou les deux, l'un validant l'autre ? Je pencherais pour les deux.
+ Revue visée (AJP ou EJP) et langue (anglais).
+ Qui fait la recherche bibliographique approfondie (Google Scholar, archives AJP/EJP) ?

#heading(numbering: none)[Annexe : reproduction]
Tout est dans le répertoire de travail de l'agent (`scratch/work/`), environnement `uv` avec JAX (GPU), diffrax et matplotlib. `electro.py` est une copie verbatim des cellules du notebook ; `electro_yee.py` sa variante sur grille décalée ; `gridrun.py` et `compare.py` font la comparaison avec le solveur exact ; `sheets.py` est le solveur de référence et `common.py` contient les milieux et les impulsions ; chaque figure a son script (`fig_*.py`). Rien n'a été poussé sur le dépôt de Mario.

#bibliography("refs.bib", title: [Références (à vérifier avant soumission)], style: "american-physics-society")
