# Data Quality Checker

Outil d'analyse de qualité des données conçu pour les profils 
métiers et techniques.
Analysez n'importe quel fichier CSV ou Excel en quelques secondes 
et obtenez un rapport complet de qualité.

## Fonctionnalités

### Analyse qualité
- **Complétude** — détection des données manquantes par colonne
- **Unicité** — détection des doublons et colonnes à contrainte unique
- **Cohérence** — outliers, formats mixtes, espaces superflus, casse incohérente
- **Feux tricolores** — niveau de criticité immédiatement visible
- **Recommandations métier** — actions concrètes en langage non-technique
- **Score de qualité global** — pondéré complétude/unicité/cohérence

### Multi-fichiers et croisement
- Import simultané de plusieurs fichiers CSV ou Excel
- Fusion (merge/join) de deux fichiers sur une colonne clé commune
- Choix du type de jointure : Left, Inner, Outer

### Gouvernance et cartographie
- **Cartographie enrichie** : types détectés, cardinalité, échantillon de 5 valeurs, taille mémoire par colonne
- **Dictionnaire de données** éditable (Data Owner, RGPD, règles de gestion)
- **Export Excel de gouvernance** à 3-4 feuilles (Accueil, Technique, Fonctionnel, Lignage)
- **Rapport PDF** structuré avec page de garde, résumé exécutif et lignage

### Lignage des données (Data Lineage)
- Traçage automatique de l'historique des actions (import, fusion, export)
- Timeline visuelle dans l'interface
- Export du lignage en CSV et intégration dans le rapport PDF et l'Excel

### Performance
- Mise en cache des analyses (`st.cache_data`) pour les gros fichiers
- Optimisation pandas pour les fichiers de 50 000+ lignes

## Lancement

### Interface web (recommandé)
```bash
pip install -r requirements.txt
streamlit run data_quality.py
```

### Ligne de commande
```bash
python data_quality.py mon_fichier.csv
```

## Stack technique

Python · Pandas · Streamlit · ReportLab · OpenPyXL · Matplotlib · Seaborn

## Tests

```bash
pip install -r requirements.txt
pytest
```

## Architecture des modules

- `data_quality.py` — moteur d'analyse, interface Streamlit, génération PDF/Excel, CLI
- `interactive_cleaner.py` — module de nettoyage interactif (fonctions pures, testables)
- `data_governance.py` — lignage, fusion multi-fichiers, enrichissement de cartographie
