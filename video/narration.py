"""Narration of the video, one entry per section: (name, text shown in subtitles, text spoken by the TTS)."""

SECTIONS = [
    ("intro",
     "Dans le verre, la lumière semble avancer un tiers moins vite que dans le vide. C'est ce que mesure l'indice "
     "de réfraction. Pourtant, entre les atomes, il n'y a que du vide, et dans le vide la lumière va toujours à la "
     "même vitesse, c. Alors, qui ralentit ? Je vais vous montrer, simulation à l'appui, que la réponse est : "
     "personne. Ce qui est lent, c'est une somme."),
    ("model",
     "Simplifions, sans tricher. Une onde plane qui arrive de face sur une lame ne dépend que d'une seule "
     "coordonnée : x. Chaque couche d'atomes devient alors un feuillet, une nappe de charges retenues par des "
     "ressorts. Quand le champ électrique les secoue, les charges oscillent, et cette nappe de courant rayonne à son "
     "tour une onde plane, des deux côtés, à la vitesse c. En une dimension, c'est même exact : le champ émis par un "
     "feuillet vaut moins la moitié de son courant, avec un simple retard, la distance divisée par c."),
    ("sum",
     "Les équations de Maxwell sont linéaires. Le champ total est donc l'onde incidente, plus la somme des ondes "
     "émises par tous les feuillets. Chaque terme de cette somme se propage à c. Aucun ne ralentit. Gardez cette "
     "équation en tête : tout le reste en découle."),
    ("sheet",
     "Commençons par un seul feuillet. Si l'impulsion est lente devant la fréquence propre des ressorts, les charges "
     "suivent le champ presque sans délai. Leur vitesse est alors proportionnelle à la dérivée du champ, et le "
     "feuillet rayonne moins tau fois cette dérivée. Or, ajouter à une courbe un peu de sa dérivée, avec un signe "
     "moins, revient à la décaler dans le temps. Derrière le feuillet, on retrouve la même impulsion, légèrement en "
     "retard. Et le même rayonnement, émis vers l'arrière, c'est l'écho : la réflexion."),
    ("slab",
     "Empilons maintenant soixante feuillets. Chacun ajoute son petit retard, et les retards s'additionnent. Voici la "
     "simulation exacte. En blanc, le champ total : l'impulsion ralentit dans la lame, puis repart à c. En gris, "
     "l'onde incidente seule : elle continue tout droit, à c, comme si la matière n'existait pas. En orange, ce que "
     "rayonnent les charges : une copie inversée de l'onde incidente, qui l'efface exactement, plus une impulsion en "
     "retard. Voilà le théorème d'extinction d'Ewald et Oseen, en direct."),
    ("spacetime",
     "En espace-temps, c'est encore plus net. À gauche, le champ total. Au centre, l'onde incidente, qui file à c. "
     "À droite, le rayonnement de la matière, qui contient les deux impulsions : l'une inversée, à c, l'autre "
     "retardée."),
    ("front",
     "Un détail confirme tout : le front de l'impulsion. Les charges ne peuvent pas réagir avant que le champ ne les "
     "atteigne. Le tout premier bout de l'onde traverse donc la matière à c, sans le moindre retard ; c'est la suite "
     "qui se fait attendre. Et il y a deux retards différents : celui des crêtes, donné par la vitesse de phase, et "
     "celui de l'enveloppe, donné par la vitesse de groupe. Ici, l'enveloppe prend deux fois plus de retard que les "
     "crêtes."),
    ("colors",
     "Tout dépend de la fréquence. Sous la résonance des ressorts, le milieu ralentit la lumière. Près de la "
     "résonance, il la ralentit énormément, et la déforme. Juste au-dessus, plus rien ne passe : la lame devient un "
     "miroir. Et plus haut encore, l'indice devient plus petit que un : les crêtes vont plus vite que c ! Pas de "
     "panique : l'enveloppe, elle, reste plus lente que la lumière. Aucun signal ne dépasse c."),
    ("slowlight",
     "Une partie de la lumière traîne même très longtemps. Après le passage de l'impulsion, la lame continue de "
     "vibrer, de plus en plus faiblement. Ce sont les fréquences proches du bord de bande, où la vitesse de groupe "
     "tend vers zéro. L'énergie qui reste décroît comme un sur t au carré, et non comme une exponentielle."),
    ("refraction",
     "Passons en deux dimensions, avec un faisceau oblique. La lame est invariante le long de sa surface : chaque "
     "onde plane garde sa période le long de cette surface. C'est la loi de Snell-Descartes, et elle sort toute "
     "seule de la simulation : mesure et prédiction coïncident à un centième de degré près. Et regardez : là "
     "encore, l'onde incidente continue tout droit, à c. La matière l'éteint, et la remplace par le faisceau "
     "réfracté et le faisceau réfléchi."),
    ("summary",
     "Résumons. Dans un milieu, rien ne va moins vite que la lumière. L'onde incidente traverse tout à c. Les "
     "charges qu'elle met en mouvement rayonnent, elles aussi à c, une onde qui l'annule et une autre qui la "
     "remplace, en retard. L'indice de réfraction, la lumière lente, la réfraction : tout cela, c'est "
     "l'interférence d'un chœur d'oscillateurs."),
    ("credits",
     "Vidéo écrite et réalisée par Claude, à partir de la simulation de Mario Geiger et d'une idée de Quentin "
     "Talon. Merci de votre attention."),
]

# words the voice mispronounces, spelled for the ear only
SPOKEN = {"Ewald et Oseen": "Évalde et Osséne", "Geiger": "Gaïgueur"}


def spoken(text):
    for a, b in SPOKEN.items():
        text = text.replace(a, b)
    return text


# seconds of picture before and after the voice, per section
LEAD = {"intro": 3.0}
TAIL = {"credits": 3.0, "summary": 1.5}
DEFAULT_LEAD, DEFAULT_TAIL = 0.6, 1.0


def timing(durations):
    """Section -> (lead, speech, total) in seconds."""
    return {n: (LEAD.get(n, DEFAULT_LEAD), d, LEAD.get(n, DEFAULT_LEAD) + d + TAIL.get(n, DEFAULT_TAIL))
            for n, d in durations.items()}
