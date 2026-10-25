# -*- coding: utf-8 -*-
"""
============================================================
  DATA QUALITY CHECKER — Version Métier
============================================================
Application d'analyse de qualité des données conçue pour des
profils métiers (chefs de projet, responsables qualité,
directeurs) tout en gardant la rigueur technique.

Utilisation :
  - Mode CLI       : python data_quality.py mon_fichier.csv
  - Mode Streamlit : streamlit run data_quality.py

Dépendances :
  pip install pandas numpy matplotlib seaborn streamlit reportlab openpyxl
============================================================
"""

import os
import sys
import csv
import io
import re
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from data_governance import (
    DataLineage,
    merge_dataframes,
    find_common_columns,
    enrichir_cartographie,
    estimate_memory_usage,
    generer_feuille_lignage,
    export_clean_csv,
    export_clean_excel,
)

# ReportLab pour le PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# ============================================================
#  1. CHARGEMENT DES DONNÉES
# ============================================================

def detecter_separateur_csv(chemin: str) -> str:
    """Détecte automatiquement le séparateur d'un fichier CSV."""
    with open(chemin, "r", encoding="utf-8", errors="ignore") as f:
        echantillon = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(echantillon, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        # Fallback : on compte les occurrences
        candidats = [",", ";", "\t", "|"]
        comptes = {c: echantillon.count(c) for c in candidats}
        return max(comptes, key=comptes.get)


def charger_fichier(chemin: str) -> pd.DataFrame:
    """Charge un fichier CSV ou Excel et retourne un DataFrame."""
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    ext = os.path.splitext(chemin)[1].lower()
    if ext == ".csv":
        sep = detecter_separateur_csv(chemin)
        return pd.read_csv(chemin, sep=sep, encoding="utf-8",
                           encoding_errors="ignore", low_memory=False)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(chemin)
    else:
        raise ValueError(f"Format non supporté : {ext}")


def obtenir_metadonnees(chemin: str, df: pd.DataFrame) -> dict:
    """Retourne les métadonnées du fichier analysé."""
    taille_octets = os.path.getsize(chemin)
    return {
        "nom_fichier": os.path.basename(chemin),
        "taille_ko": round(taille_octets / 1024, 2),
        "taille_mo": round(taille_octets / (1024 * 1024), 3),
        "nb_lignes": int(df.shape[0]),
        "nb_colonnes": int(df.shape[1]),
        "colonnes": list(df.columns),
        "types": {c: str(df[c].dtype) for c in df.columns},
    }


# ============================================================
#  2. ANALYSES QUALITÉ
# ============================================================

def analyser_completude(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse les données manquantes par colonne."""
    lignes = []
    n = len(df) if len(df) > 0 else 1
    for col in df.columns:
        s = df[col]
        manquantes = int(s.isna().sum())
        # Cellules vides texte (chaînes vides ou espaces)
        if s.dtype == object:
            vides = int(s.fillna("").astype(str).str.strip().eq("").sum())
            manquantes = max(manquantes, vides)
        pct = round(100 * manquantes / n, 2)
        if pct == 0:
            niveau = "🟢"
        elif pct < 10:
            niveau = "🟡"
        else:
            niveau = "🔴"
        lignes.append({
            "Nom de la colonne": col,
            "Données manquantes": manquantes,
            "% de données manquantes": pct,
            "Niveau de criticité": niveau,
        })
    return pd.DataFrame(lignes)


def analyser_unicite(df: pd.DataFrame) -> dict:
    """Analyse les doublons et l'unicité."""
    doublons_lignes = int(df.duplicated().sum())
    details = []
    for col in df.columns:
        uniques = int(df[col].nunique(dropna=True))
        total = int(df[col].notna().sum()) or 1
        pct = round(100 * uniques / total, 2)
        # On suppose qu'une colonne contenant id/code/email doit être unique
        nom = col.lower()
        doit_etre_unique = any(k in nom for k in ["id", "code", "email", "mail", "uuid", "ref"])
        probleme = doit_etre_unique and uniques < total
        if not doit_etre_unique:
            niveau = "🟢"
        elif probleme:
            niveau = "🔴"
        else:
            niveau = "🟢"
        details.append({
            "Nom de la colonne": col,
            "Valeurs distinctes": uniques,
            "% de valeurs distinctes": pct,
            "Doit être unique": "Oui" if doit_etre_unique else "Non",
            "Problème d'unicité détecté": "Oui" if probleme else "Non",
            "Niveau de criticité": niveau,
        })
    return {
        "doublons_lignes": doublons_lignes,
        "details": pd.DataFrame(details),
    }


def _est_colonne_date(serie: pd.Series) -> bool:
    """Heuristique : la colonne contient-elle des dates ?"""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return True
    if serie.dtype != object:
        return False
    ech = serie.dropna().astype(str).head(20)
    if len(ech) == 0:
        return False
    pattern = re.compile(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}")
    hits = sum(1 for v in ech if pattern.search(v))
    return hits >= max(3, len(ech) // 2)


def analyser_coherence(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse la cohérence : outliers, formats, espaces, casse, valeurs négatives."""
    lignes = []
    for col in df.columns:
        s = df[col]
        outliers = 0
        dates_mixtes = False
        espaces = 0
        casse = False
        negatifs = 0

        # Outliers numériques (IQR)
        if pd.api.types.is_numeric_dtype(s):
            sn = s.dropna()
            if len(sn) > 4:
                q1, q3 = sn.quantile(0.25), sn.quantile(0.75)
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = int(((sn < low) | (sn > high)).sum())
            # Valeurs négatives anormales si le nom suggère un montant/âge/quantité
            nom = col.lower()
            if any(k in nom for k in ["prix", "price", "amount", "montant", "age", "quantite", "quantity", "stock"]):
                negatifs = int((sn < 0).sum())

        # Colonnes texte : espaces, casse, dates mixtes
        if s.dtype == object:
            txt = s.dropna().astype(str)
            espaces = int((txt != txt.str.strip()).sum())
            if len(txt) > 0:
                low = (txt.str.islower()).sum()
                up = (txt.str.isupper()).sum()
                cap = (txt.str.istitle()).sum()
                # Si plusieurs styles coexistent → casse incohérente
                styles = sum(1 for v in [low, up, cap] if v > 0)
                casse = styles >= 2 and len(txt) > 5

            if _est_colonne_date(s):
                formats = set()
                for v in txt.head(50):
                    if re.search(r"\d{4}-\d{2}-\d{2}", v):
                        formats.add("ISO")
                    elif re.search(r"\d{2}/\d{2}/\d{4}", v):
                        formats.add("FR")
                    elif re.search(r"\d{2}-\d{2}-\d{4}", v):
                        formats.add("FR-")
                    elif re.search(r"\d{4}/\d{2}/\d{2}", v):
                        formats.add("ISO/")
                dates_mixtes = len(formats) >= 2

        probleme_critique = dates_mixtes or negatifs > 0
        probleme_mineur = outliers > 0 or espaces > 0 or casse
        if probleme_critique:
            niveau = "🔴"
        elif probleme_mineur:
            niveau = "🟡"
        else:
            niveau = "🟢"

        lignes.append({
            "Nom de la colonne": col,
            "Valeurs aberrantes": outliers,
            "Formats de dates incohérents": "Oui" if dates_mixtes else "Non",
            "Espaces superflus détectés": espaces,
            "Majuscules/minuscules incohérentes": "Oui" if casse else "Non",
            "Valeurs négatives anormales": negatifs,
            "Niveau de criticité": niveau,
        })
    return pd.DataFrame(lignes)


# ============================================================
#  3. SCORES
# ============================================================

def lancer_analyse_complete(df: pd.DataFrame) -> dict:
    """Enchaîne toutes les analyses et renvoie un dict avec tous les résultats.

    Fonction utilitaire pure (pas de Streamlit, pas de chemin de fichier).
    Permet de recalculer facilement le score après nettoyage ou fusion.
    """
    completude = analyser_completude(df)
    unicite = analyser_unicite(df)
    coherence = analyser_coherence(df)
    scores = calculer_scores(df, completude, unicite, coherence)
    recos = generer_recommandations(df, completude, unicite, coherence)
    return {
        "completude": completude,
        "unicite": unicite,
        "coherence": coherence,
        "scores": scores,
        "recos": recos,
    }


def calculer_scores(df: pd.DataFrame, completude: pd.DataFrame,
                    unicite: dict, coherence: pd.DataFrame) -> dict:
    """Calcule les scores de qualité (0 à 100)."""
    # Complétude
    pct_moy_manquantes = completude["% de données manquantes"].mean() if len(completude) else 0
    score_completude = max(0, 100 - pct_moy_manquantes)

    # Unicité
    n = max(1, len(df))
    pct_doublons = 100 * unicite["doublons_lignes"] / n
    pb_unicite = (unicite["details"]["Problème d'unicité détecté"] == "Oui").sum()
    score_unicite = max(0, 100 - pct_doublons - 10 * pb_unicite)

    # Cohérence
    pb_critiques = ((coherence["Formats de dates incohérents"] == "Oui").sum()
                    + (coherence["Valeurs négatives anormales"] > 0).sum())
    pb_mineurs = ((coherence["Valeurs aberrantes"] > 0).sum()
                  + (coherence["Espaces superflus détectés"] > 0).sum()
                  + (coherence["Majuscules/minuscules incohérentes"] == "Oui").sum())
    score_coherence = max(0, 100 - 10 * pb_critiques - 3 * pb_mineurs)

    score_global = round(
        0.40 * score_completude + 0.30 * score_unicite + 0.30 * score_coherence, 1
    )
    return {
        "Score de complétude": round(score_completude, 1),
        "Score d'unicité": round(score_unicite, 1),
        "Score de cohérence": round(score_coherence, 1),
        "Score de qualité global": score_global,
    }


def niveau_qualite(score: float) -> tuple:
    """Retourne (label, couleur_hex, emoji) selon le score."""
    if score >= 80:
        return ("BONNE", "#16a34a", "✅")
    if score >= 60:
        return ("MOYENNE", "#f59e0b", "⚠️")
    return ("CRITIQUE", "#dc2626", "🚨")


# Mapping des emojis vers du texte PDF-safe (Helvetica n'a pas les glyphes emoji)
_EMOJI_PDF = {
    "🟢": "[OK] ", "🟡": "[!] ", "🔴": "[X] ",
    "✅": "[OK] ", "⚠️": "[!] ", "🚨": "[X] ",
    "🎉": "", "📊": "", "📌": "", "📋": "", "📁": "",
    "🔁": "", "🔍": "", "💡": "", "🎯": "", "🔝": "",
    "➡️": "-> ", "⬇️": "", "📄": "", "👆": "",
}

def _safe_pdf(txt: str) -> str:
    """Nettoie une chaîne pour le PDF (remplace les emojis non rendus)."""
    if not isinstance(txt, str):
        return txt
    for e, r in _EMOJI_PDF.items():
        txt = txt.replace(e, r)
    return txt


def _couleur_niveau(niveau_emoji: str):
    """Retourne la couleur ReportLab associée à un niveau (emoji)."""
    if niveau_emoji == "🔴":
        return colors.HexColor("#dc2626")
    if niveau_emoji == "🟡":
        return colors.HexColor("#f59e0b")
    return colors.HexColor("#16a34a")



# ============================================================
#  4. RECOMMANDATIONS MÉTIER CONTEXTUALISÉES
# ============================================================

def _impact_metier(col: str) -> str:
    """Retourne l'impact métier en fonction du nom de la colonne."""
    nom = col.lower()
    if any(k in nom for k in ["email", "mail"]):
        return ("Ces clients ne peuvent pas être contactés par email, "
                "ce qui représente un manque à gagner potentiel et limite "
                "les campagnes de communication.")
    if any(k in nom for k in ["id", "code", "uuid", "ref"]):
        return ("Sans identifiant fiable, la traçabilité et les jointures "
                "avec d'autres bases de données sont compromises.")
    if "date" in nom or "jour" in nom:
        return ("Les analyses temporelles (tendances, saisonnalité, cohortes) "
                "deviennent imprécises voire impossibles.")
    if any(k in nom for k in ["prix", "price", "montant", "amount", "ca", "chiffre"]):
        return ("Les calculs financiers (chiffre d'affaires, marges, prévisions) "
                "seront faussés.")
    if any(k in nom for k in ["nom", "prenom", "name", "client"]):
        return ("L'identification des clients devient difficile, "
                "ce qui dégrade la relation commerciale et le suivi.")
    if any(k in nom for k in ["tel", "phone", "mobile"]):
        return ("Impossible de joindre ces contacts par téléphone, "
                "perte d'opportunités commerciales.")
    return ("La fiabilité globale des analyses et des décisions basées "
            "sur ces données est dégradée.")


def generer_recommandations(df: pd.DataFrame, completude: pd.DataFrame,
                            unicite: dict, coherence: pd.DataFrame) -> list:
    """Génère une liste de recommandations contextualisées triées par priorité."""
    recos = []
    n = len(df)

    # Complétude
    for _, r in completude.iterrows():
        if r["Données manquantes"] > 0:
            col = r["Nom de la colonne"]
            pct = r["% de données manquantes"]
            niveau = r["Niveau de criticité"]
            priorite = 0 if niveau == "🔴" else (1 if niveau == "🟡" else 2)
            recos.append({
                "priorite": priorite,
                "niveau": niveau,
                "titre": "DONNÉES MANQUANTES",
                "colonne": col,
                "constat": f"{int(r['Données manquantes'])} valeurs manquantes sur {n} lignes ({pct}%)",
                "impact": _impact_metier(col),
                "action": ("Vérifier la source de collecte de ces données et "
                           "compléter les valeurs manquantes, ou décider d'exclure "
                           "ces lignes des analyses."),
            })

    # Unicité — doublons globaux
    if unicite["doublons_lignes"] > 0:
        recos.append({
            "priorite": 0,
            "niveau": "🔴",
            "titre": "LIGNES EN DOUBLON",
            "colonne": "(toutes colonnes)",
            "constat": f"{unicite['doublons_lignes']} lignes strictement identiques détectées",
            "impact": ("Les doublons faussent tous les comptages, "
                       "agrégats et indicateurs de pilotage."),
            "action": "Dédoublonner le fichier avant toute analyse ou import en base.",
        })

    # Unicité — colonnes qui devraient être uniques
    for _, r in unicite["details"].iterrows():
        if r["Problème d'unicité détecté"] == "Oui":
            col = r["Nom de la colonne"]
            recos.append({
                "priorite": 0,
                "niveau": "🔴",
                "titre": "PROBLÈME D'UNICITÉ",
                "colonne": col,
                "constat": f"La colonne '{col}' contient des valeurs en double alors qu'elle devrait être unique",
                "impact": _impact_metier(col),
                "action": ("Identifier les doublons et conserver la valeur la plus "
                           "récente ou la plus complète."),
            })

    # Cohérence
    for _, r in coherence.iterrows():
        col = r["Nom de la colonne"]
        if r["Formats de dates incohérents"] == "Oui":
            recos.append({
                "priorite": 0, "niveau": "🔴",
                "titre": "FORMATS DE DATES INCOHÉRENTS",
                "colonne": col,
                "constat": "Plusieurs formats de dates coexistent dans la même colonne",
                "impact": _impact_metier(col),
                "action": "Standardiser toutes les dates au format ISO (AAAA-MM-JJ).",
            })
        if r["Valeurs négatives anormales"] > 0:
            recos.append({
                "priorite": 0, "niveau": "🔴",
                "titre": "VALEURS NÉGATIVES ANORMALES",
                "colonne": col,
                "constat": f"{int(r['Valeurs négatives anormales'])} valeurs négatives détectées",
                "impact": _impact_metier(col),
                "action": "Vérifier la saisie ; corriger ou exclure ces lignes.",
            })
        if r["Valeurs aberrantes"] > 0:
            recos.append({
                "priorite": 1, "niveau": "🟡",
                "titre": "VALEURS ABERRANTES",
                "colonne": col,
                "constat": f"{int(r['Valeurs aberrantes'])} valeurs hors plage normale (méthode IQR)",
                "impact": "Les moyennes et tendances peuvent être faussées par ces extrêmes.",
                "action": "Examiner ces valeurs : erreur de saisie ou cas exceptionnel légitime ?",
            })
        if r["Espaces superflus détectés"] > 0:
            recos.append({
                "priorite": 1, "niveau": "🟡",
                "titre": "ESPACES SUPERFLUS",
                "colonne": col,
                "constat": f"{int(r['Espaces superflus détectés'])} valeurs contiennent des espaces en début/fin",
                "impact": "Provoque des doublons cachés et fausse les jointures.",
                "action": "Nettoyer les espaces (trim) sur cette colonne.",
            })
        if r["Majuscules/minuscules incohérentes"] == "Oui":
            recos.append({
                "priorite": 1, "niveau": "🟡",
                "titre": "CASSE INCOHÉRENTE",
                "colonne": col,
                "constat": "Mélange de majuscules et minuscules dans la même colonne",
                "impact": "Peut créer des doublons (ex : 'Paris' vs 'PARIS' vs 'paris').",
                "action": "Standardiser la casse (tout en minuscules ou en Title Case).",
            })

    recos.sort(key=lambda x: x["priorite"])
    return recos


def top_3_problemes(recos: list) -> list:
    """Renvoie les 3 problèmes prioritaires formulés simplement."""
    top = []
    for r in recos[:3]:
        top.append(f"{r['niveau']} {r['titre']} sur '{r['colonne']}' — {r['constat']}")
    return top


# ============================================================
#  4b. GOUVERNANCE & CARTOGRAPHIE DES DONNÉES
# ============================================================

def _detecter_format_colonne(s: pd.Series) -> str:
    """Détecte automatiquement le format d'une colonne (Date, Email, Téléphone, Code postal, Numérique, Texte libre)."""
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return "Texte libre"
        
    if pd.api.types.is_numeric_dtype(s):
        return "Numérique"
        
    if pd.api.types.is_datetime64_any_dtype(s):
        return "Date"
        
    # Échantillonnage
    echantillon = s_clean.astype(str).str.strip().head(50)
    
    # Regexes de détection
    email_pat = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
    tel_pat = re.compile(r"^(?:\+33|0033|0)[1-9](?:[\s.-]?\d{2}){4}$")
    cp_pat = re.compile(r"^\d{5}$")
    date_pat = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$")
    
    nb_emails = sum(1 for v in echantillon if email_pat.match(v))
    nb_tels = sum(1 for v in echantillon if tel_pat.match(v))
    nb_cps = sum(1 for v in echantillon if cp_pat.match(v))
    nb_dates = sum(1 for v in echantillon if date_pat.match(v))
    
    total = len(echantillon)
    if nb_emails >= max(1, total // 3):
        return "Email"
    if nb_tels >= max(1, total // 3):
        return "Téléphone"
    if nb_cps >= max(1, total // 3):
        return "Code postal"
    if nb_dates >= max(1, total // 3):
        return "Date"
        
    return "Texte libre"


def _formater_type_technique(df: pd.DataFrame, col: str) -> str:
    """Traduit le type technique d'une colonne Pandas en termes métier."""
    s = df[col]
    dtype_str = str(s.dtype)
    if "int" in dtype_str:
        return "Nombre entier"
    elif "float" in dtype_str:
        return "Nombre décimal"
    elif "bool" in dtype_str:
        return "Booléen"
    elif "datetime" in dtype_str:
        return "Date"
    elif dtype_str == "object":
        if _est_colonne_date(s):
            return "Date"
        return "Texte"
    return "Autre"


def extraire_metadonnees_cartographie(df: pd.DataFrame, completude: pd.DataFrame,
                                      unicite: dict, coherence: pd.DataFrame) -> pd.DataFrame:
    """Génère un DataFrame combinant métadonnées techniques et structure fonctionnelle pour la cartographie."""
    rows = []
    for col in df.columns:
        # Type technique
        type_tech = _formater_type_technique(df, col)
        # Format détecté
        fmt = _detecter_format_colonne(df[col])
        
        # Complétude
        comp_row = completude[completude["Nom de la colonne"] == col]
        pct_manquant = float(comp_row["% de données manquantes"].iloc[0]) if not comp_row.empty else 0.0
        pct_completude = round(100.0 - pct_manquant, 1)
        
        # Unicité / Valeurs distinctes
        uni_row = unicite["details"][unicite["details"]["Nom de la colonne"] == col]
        val_distinctes = int(uni_row["Valeurs distinctes"].iloc[0]) if not uni_row.empty else 0
        
        # Longueurs min/max pour texte
        lg_min, lg_max = "—", "—"
        if type_tech == "Texte":
            lengths = df[col].dropna().astype(str).str.len()
            if not lengths.empty:
                lg_min = int(lengths.min())
                lg_max = int(lengths.max())
                
        # Plages de valeurs min/max pour numérique
        val_min, val_max = "—", "—"
        if pd.api.types.is_numeric_dtype(df[col]):
            sn = df[col].dropna()
            if not sn.empty:
                val_min = round(float(sn.min()), 2)
                val_max = round(float(sn.max()), 2)
                
        # Valeur la plus fréquente
        val_freq, freq_count = "—", 0
        vc = df[col].dropna().value_counts()
        if not vc.empty:
            val_freq = str(vc.index[0])
            freq_count = int(vc.iloc[0])
            
        # Diagnostic Qualité
        coh_row = coherence[coherence["Nom de la colonne"] == col]
        anomalies_list = []
        if not coh_row.empty:
            r_coh = coh_row.iloc[0]
            if r_coh["Formats de dates incohérents"] == "Oui":
                anomalies_list.append("dates mixtes")
            if r_coh["Valeurs négatives anormales"] > 0:
                anomalies_list.append("négatifs")
            if r_coh["Valeurs aberrantes"] > 0:
                anomalies_list.append(f"{int(r_coh['Valeurs aberrantes'])} outliers")
            if r_coh["Espaces superflus détectés"] > 0:
                anomalies_list.append("espaces superflus")
            if r_coh["Majuscules/minuscules incohérentes"] == "Oui":
                anomalies_list.append("casse incohérente")
        
        alertes_qualite = "🟢 Aucune anomalie" if not anomalies_list else "🔴 " + ", ".join(anomalies_list)

        rows.append({
            # Métadonnées Techniques
            "Nom de la colonne": col,
            "Type technique": type_tech,
            "Format détecté": fmt,
            "Complétude (%)": pct_completude,
            "Valeurs distinctes": val_distinctes,
            "Longueur min": lg_min,
            "Longueur max": lg_max,
            "Valeur min": val_min,
            "Valeur max": val_max,
            "Valeur la plus fréquente": val_freq,
            "Occurrences": freq_count,
            "Diagnostic Qualité": alertes_qualite,
            # Métadonnées Fonctionnelles
            "Description Fonctionnelle": "",
            "Propriétaire de la donnée (Data Owner)": "",
            "Criticité Métier": "Faible",
            "Sensibilité RGPD (Donnée Personnelle)": "Non",
            "Source de la donnée": "ERP",
            "Fréquence de mise à jour": "Quotidienne",
            "Définition métier officielle": "",
            "Équipes consommatrices": "",
            "Lien avec d'autres données": ""
        })
        
    return pd.DataFrame(rows)


def fusionner_cartographie_existante(df_nouvelle: pd.DataFrame, sheet_dict: dict) -> pd.DataFrame:
    """Fusionne la nouvelle cartographie technique avec les métadonnées fonctionnelles d'une cartographie précédente (3 feuilles)."""
    # Recherche de la feuille fonctionnelle ou dictionnaire
    sheet_name = None
    for s in sheet_dict.keys():
        if "fonctionnel" in s.lower():
            sheet_name = s
            break
    if not sheet_name:
        for s in sheet_dict.keys():
            if "dictionnaire" in s.lower():
                sheet_name = s
                break
    if not sheet_name:
        for s in sheet_dict.keys():
            if "accueil" not in s.lower() and "synthèse" not in s.lower():
                sheet_name = s
                break
    if not sheet_name:
        sheet_name = list(sheet_dict.keys())[0]
        
    df_existante = sheet_dict[sheet_name].copy()
    
    # Nettoyage des colonnes
    df_existante.columns = [str(c).strip() for c in df_existante.columns]
    
    # Si la ligne d'en-tête est décalée ou masquée
    if "Nom de la colonne" not in df_existante.columns:
        header_row_idx = None
        for i, row in enumerate(df_existante.values):
            if any(str(x).strip() == "Nom de la colonne" for x in row):
                header_row_idx = i
                break
        if header_row_idx is not None:
            new_cols = [str(x).strip() for x in df_existante.iloc[header_row_idx]]
            df_existante = df_existante.iloc[header_row_idx + 1:]
            df_existante.columns = new_cols
            
    col_nom = None
    for c in ["Nom de la colonne", "Nom colonne", "Colonne", "Technical Name", "Column Name", "Nom"]:
        if c in df_existante.columns:
            col_nom = c
            break
            
    if not col_nom:
        return df_nouvelle
        
    # Mapper toutes les colonnes fonctionnelles (anciennes et nouvelles)
    fonctionnel_mapping = {
        "Description Fonctionnelle": ["Description Fonctionnelle", "Description", "Definition", "Définition", "Description fonctionnelle"],
        "Propriétaire de la donnée (Data Owner)": ["Propriétaire de la donnée (Data Owner)", "Propriétaire", "Owner", "Data Owner", "Propriétaire de la donnée"],
        "Criticité Métier": ["Criticité Métier", "Criticité", "Criticality", "Importance"],
        "Sensibilité RGPD (Donnée Personnelle)": ["Sensibilité RGPD (Donnée Personnelle)", "RGPD", "GDPR", "Sensibilité", "Sensible", "Donnée Personnelle"],
        "Source de la donnée": ["Source de la donnée", "Source", "Source de la donnée", "Data Source"],
        "Fréquence de mise à jour": ["Fréquence de mise à jour", "Fréquence", "Update Frequency", "Frequency"],
        "Définition métier officielle": ["Définition métier officielle", "Définition officielle", "Définition métier", "Official business definition", "Business Definition"],
        "Équipes consommatrices": ["Équipes consommatrices", "Équipes", "Consommateurs", "Consumer teams", "Consumers"],
        "Lien avec d'autres données": ["Lien avec d'autres données", "Lien", "Liens", "Lien avec d'autres tables", "Links"]
    }
    
    df_existante.set_index(col_nom, inplace=True)
    df_existante = df_existante[~df_existante.index.duplicated(keep='first')]
    
    for dest_col, source_candidates in fonctionnel_mapping.items():
        cand_trouve = None
        for c in source_candidates:
            if c in df_existante.columns:
                cand_trouve = c
                break
        if cand_trouve:
            def get_val(nom_col):
                if nom_col in df_existante.index:
                    val = df_existante.loc[nom_col, cand_trouve]
                    if pd.notna(val) and str(val).strip() != "":
                        return str(val).strip()
                # Conserver la valeur courante de la nouvelle s'il y a déjà une valeur
                old_val = df_nouvelle.loc[df_nouvelle["Nom de la colonne"] == nom_col, dest_col]
                return old_val.iloc[0] if not old_val.empty else ""
            df_nouvelle[dest_col] = df_nouvelle["Nom de la colonne"].apply(get_val)
            
    return df_nouvelle


def generer_excel_cartographie(df_carto: pd.DataFrame, nom_fichier_origine: str,
                              lineage: DataLineage | None = None) -> bytes:
    """Génère un fichier Excel de cartographie premium à 3-4 feuilles avec styles, KPI et listes déroulantes.

    Si un objet DataLineage est fourni, une feuille « Lignage des Données » est ajoutée.
    """
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.worksheet.datavalidation import DataValidation
    
    wb = openpyxl.Workbook()
    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])
        
    font_titre = Font(name="Segoe UI", size=16, bold=True, color="FFFFFF")
    font_h2 = Font(name="Segoe UI", size=12, bold=True, color="0F172A")
    font_bold = Font(name="Segoe UI", size=10, bold=True, color="0F172A")
    font_normal = Font(name="Segoe UI", size=10, color="0F172A")
    font_kpi_num = Font(name="Segoe UI", size=20, bold=True, color="1E3A8A")
    font_kpi_label = Font(name="Segoe UI", size=9, bold=True, color="475569")
    
    fill_deep_blue = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    fill_tech_header = PatternFill(start_color="1E40AF", end_color="1E40AF", fill_type="solid")
    fill_func_header = PatternFill(start_color="065F46", end_color="065F46", fill_type="solid")
    fill_kpi = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
    fill_zebra_light = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    fill_zebra_white = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
    fill_alert_red = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    fill_alert_green = PatternFill(start_color="D1FAE5", end_color="D1FAE5", fill_type="solid")
    
    # Vert clair pastel pour indiquer les champs éditables à remplir
    fill_input_green = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")
    
    border_thin = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='CBD5E1'),
        bottom=Side(style='thin', color='CBD5E1')
    )
    
    # ============================================================
    #  FEUILLE 1 : ACCUEIL & SYNTHÈSE
    # ============================================================
    ws_accueil = wb.create_sheet(title="Accueil & Synthèse")
    ws_accueil.views.sheetView[0].showGridLines = True
    
    ws_accueil.merge_cells("A1:H2")
    cell_titre = ws_accueil["A1"]
    cell_titre.value = "  CARTOGRAPHIE & GOUVERNANCE DES DONNÉES"
    cell_titre.font = font_titre
    cell_titre.fill = fill_deep_blue
    cell_titre.alignment = Alignment(vertical="center", horizontal="left")
    
    ws_accueil["A4"] = "Fichier d'origine :"
    ws_accueil["A4"].font = font_bold
    ws_accueil["B4"] = nom_fichier_origine
    ws_accueil["B4"].font = font_normal
    
    ws_accueil["A5"] = "Date de génération :"
    ws_accueil["A5"].font = font_bold
    ws_accueil["B5"] = datetime.now().strftime("%d/%m/%Y à %H:%M")
    ws_accueil["B5"].font = font_normal
    
    nb_colonnes = len(df_carto)
    avg_comp = df_carto["Complétude (%)"].mean() if nb_colonnes > 0 else 0
    nb_anomalies = sum(1 for x in df_carto["Diagnostic Qualité"] if "🔴" in str(x))
    nb_doc = sum(1 for x in df_carto["Description Fonctionnelle"] if x and str(x).strip() != "")
    pct_doc = round(100 * nb_doc / nb_colonnes, 1) if nb_colonnes > 0 else 0
    
    # KPIs
    ws_accueil.merge_cells("A7:B7")
    ws_accueil["A7"] = "TOTAL COLONNES"
    ws_accueil["A7"].font = font_kpi_label
    ws_accueil["A7"].fill = fill_kpi
    ws_accueil["A7"].alignment = Alignment(horizontal="center", vertical="center")
    ws_accueil.merge_cells("A8:B9")
    ws_accueil["A8"] = nb_colonnes
    ws_accueil["A8"].font = font_kpi_num
    ws_accueil["A8"].fill = fill_kpi
    ws_accueil["A8"].alignment = Alignment(horizontal="center", vertical="center")
    
    ws_accueil.merge_cells("C7:D7")
    ws_accueil["C7"] = "COMPLÉTUDE MOYENNE"
    ws_accueil["C7"].font = font_kpi_label
    ws_accueil["C7"].fill = fill_kpi
    ws_accueil["C7"].alignment = Alignment(horizontal="center", vertical="center")
    ws_accueil.merge_cells("C8:D9")
    ws_accueil["C8"] = f"{avg_comp:.1f}%"
    ws_accueil["C8"].font = font_kpi_num
    ws_accueil["C8"].fill = fill_kpi
    ws_accueil["C8"].alignment = Alignment(horizontal="center", vertical="center")
    
    ws_accueil.merge_cells("E7:F7")
    ws_accueil["E7"] = "COLONNES AVEC ANOMALIES"
    ws_accueil["E7"].font = font_kpi_label
    ws_accueil["E7"].fill = fill_kpi
    ws_accueil["E7"].alignment = Alignment(horizontal="center", vertical="center")
    ws_accueil.merge_cells("E8:F9")
    ws_accueil["E8"] = nb_anomalies
    ws_accueil["E8"].font = Font(name="Segoe UI", size=20, bold=True, color="B91C1C" if nb_anomalies > 0 else "1E3A8A")
    ws_accueil["E8"].fill = fill_kpi
    ws_accueil["E8"].alignment = Alignment(horizontal="center", vertical="center")
    
    ws_accueil.merge_cells("G7:H7")
    ws_accueil["G7"] = "COLONNES DOCUMENTÉES"
    ws_accueil["G7"].font = font_kpi_label
    ws_accueil["G7"].fill = fill_kpi
    ws_accueil["G7"].alignment = Alignment(horizontal="center", vertical="center")
    ws_accueil.merge_cells("G8:H9")
    ws_accueil["G8"] = f"{pct_doc:.1f}%"
    ws_accueil["G8"].font = Font(name="Segoe UI", size=20, bold=True, color="047857" if pct_doc >= 80 else "B45309")
    ws_accueil["G8"].fill = fill_kpi
    ws_accueil["G8"].alignment = Alignment(horizontal="center", vertical="center")
    
    for r in range(7, 10):
        for c in range(1, 9):
            ws_accueil.cell(row=r, column=c).border = border_thin
            
    ws_accueil["A11"] = "Règles & Principes de Gouvernance des Données"
    ws_accueil["A11"].font = font_h2
    
    regles = [
        ("1. Responsabilité (Data Ownership)", "Chaque colonne doit avoir un propriétaire désigné (Data Owner) responsable de la qualité et des autorisations d'accès."),
        ("2. Documentation Métier (Dictionnaire)", "Toutes les variables clés doivent être accompagnées d'une Description Fonctionnelle pour éviter les erreurs d'interprétation."),
        ("3. Classification de la Sensibilité (RGPD)", "Les colonnes contenant des données personnelles (noms, téléphones, emails, etc.) doivent être taguées comme RGPD = 'Oui' et faire l'objet de mesures de protection accrues."),
        ("4. Gestion de la Qualité en continu", "Toute colonne avec un Diagnostic Qualité en rouge (anomalie détectée) doit être corrigée à la source (système d'origine) avant d'effectuer de nouveaux imports."),
        ("5. Standardisation des Types", "Assurez-vous de la cohérence des formats (ex: formats de date homogènes, casse de texte normalisée) avant de réaliser des jointures de tables.")
    ]
    
    ligne_courante = 13
    for titre_r, desc_r in regles:
        ws_accueil.cell(row=ligne_courante, column=1, value=titre_r).font = font_bold
        ws_accueil.cell(row=ligne_courante, column=1).alignment = Alignment(vertical="top")
        ws_accueil.merge_cells(start_row=ligne_courante, start_column=2, end_row=ligne_courante+1, end_column=8)
        ws_accueil.cell(row=ligne_courante, column=2, value=desc_r).font = font_normal
        ws_accueil.cell(row=ligne_courante, column=2).alignment = Alignment(wrap_text=True, vertical="top")
        ligne_courante += 3
        
    for col_l in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        ws_accueil.column_dimensions[col_l].width = 20
    ws_accueil.column_dimensions["A"].width = 25
    
    # ============================================================
    #  FEUILLE 2 : MÉTADONNÉES TECHNIQUES
    # ============================================================
    ws_tech = wb.create_sheet(title="Métadonnées Techniques")
    ws_tech.views.sheetView[0].showGridLines = True
    
    ws_tech.merge_cells("A1:L1")
    ws_tech["A1"] = "MÉTADONNÉES TECHNIQUES EXTRAITES AUTOMATIQUEMENT"
    ws_tech["A1"].font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    ws_tech["A1"].fill = fill_tech_header
    ws_tech["A1"].alignment = Alignment(horizontal="center", vertical="center")
    
    tech_headers = [
        "Nom de la colonne", "Type technique", "Format détecté", "Complétude (%)", "Valeurs distinctes", 
        "Longueur min", "Longueur max", "Valeur min", "Valeur max", "Valeur la plus fréquente", "Occurrences", "Diagnostic Qualité"
    ]
    
    for c_idx, h in enumerate(tech_headers, 1):
        cell = ws_tech.cell(row=2, column=c_idx, value=h)
        cell.font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
        cell.fill = fill_tech_header
        cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        cell.border = border_thin
        
    cols_tech = [
        "Nom de la colonne", "Type technique", "Format détecté", "Complétude (%)", "Valeurs distinctes", 
        "Longueur min", "Longueur max", "Valeur min", "Valeur max", "Valeur la plus fréquente", "Occurrences", "Diagnostic Qualité"
    ]
    
    for r_idx, row_data in enumerate(df_carto[cols_tech].itertuples(index=False), 3):
        zebra_fill = fill_zebra_light if r_idx % 2 == 0 else fill_zebra_white
        for c_idx, val in enumerate(row_data, 1):
            cell = ws_tech.cell(row=r_idx, column=c_idx)
            cell.value = val
            cell.font = font_normal
            cell.fill = zebra_fill
            cell.border = border_thin
            
            # Alignements et formats
            if c_idx in (1, 3, 10):
                cell.alignment = Alignment(horizontal="left", vertical="center")
            elif c_idx in (2, 12):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif c_idx == 4:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = '0.0"%"'
            else:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                if isinstance(val, (int, float)):
                    cell.number_format = '#,##0' if c_idx in (5, 6, 7, 11) else '0.00'
                    
            if c_idx == 12:
                if "🔴" in str(val):
                    cell.fill = fill_alert_red
                    cell.font = Font(name="Segoe UI", size=9, bold=True, color="991B1B")
                else:
                    cell.fill = fill_alert_green
                    cell.font = Font(name="Segoe UI", size=9, bold=True, color="065F46")
                    
    ws_tech.row_dimensions[1].height = 25
    ws_tech.row_dimensions[2].height = 30
    for r in range(3, nb_colonnes + 3):
        ws_tech.row_dimensions[r].height = 22
    ws_tech.auto_filter.ref = f"A2:L{nb_colonnes+2}"
    
    largeurs_tech = {
        "A": 22, "B": 15, "C": 15, "D": 15, "E": 18, "F": 12, "G": 12, "H": 12, "I": 12, "J": 22, "K": 12, "L": 35
    }
    for col_lettre, l in largeurs_tech.items():
        ws_tech.column_dimensions[col_lettre].width = l
        
    # ============================================================
    #  FEUILLE 3 : DICTIONNAIRE FONCTIONNEL
    # ============================================================
    ws_func = wb.create_sheet(title="Dictionnaire Fonctionnel")
    ws_func.views.sheetView[0].showGridLines = True
    
    ws_func.merge_cells("A1:J1")
    ws_func["A1"] = "DICTIONNAIRE FONCTIONNEL ET GOUVERNANCE (À SAISIR — EN VERT)"
    ws_func["A1"].font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    ws_func["A1"].fill = fill_func_header
    ws_func["A1"].alignment = Alignment(horizontal="center", vertical="center")
    
    func_headers = [
        "Nom de la colonne", "Description Fonctionnelle", "Propriétaire de la donnée (Data Owner)", "Criticité Métier", 
        "Sensibilité RGPD (Donnée Personnelle)", "Source de la donnée", "Fréquence de mise à jour", 
        "Définition métier officielle", "Équipes consommatrices", "Lien avec d'autres données"
    ]
    
    for c_idx, h in enumerate(func_headers, 1):
        cell = ws_func.cell(row=2, column=c_idx, value=h)
        cell.font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
        cell.fill = fill_func_header
        cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        cell.border = border_thin
        
    cols_func = [
        "Nom de la colonne", "Description Fonctionnelle", "Propriétaire de la donnée (Data Owner)", "Criticité Métier", 
        "Sensibilité RGPD (Donnée Personnelle)", "Source de la donnée", "Fréquence de mise à jour", 
        "Définition métier officielle", "Équipes consommatrices", "Lien avec d'autres données"
    ]
    
    for r_idx, row_data in enumerate(df_carto[cols_func].itertuples(index=False), 3):
        for c_idx, val in enumerate(row_data, 1):
            cell = ws_func.cell(row=r_idx, column=c_idx)
            cell.value = val
            cell.font = font_normal
            cell.border = border_thin
            
            if c_idx == 1:
                # Nom de la colonne -> Style normal en lecture seule
                cell.fill = fill_zebra_light if r_idx % 2 == 0 else fill_zebra_white
                cell.alignment = Alignment(horizontal="left", vertical="center")
            else:
                # Champs fonctionnels métiers -> Saisie libre identifiée par fond vert clair !
                cell.fill = fill_input_green
                if c_idx in (2, 3, 8, 9, 10):
                    cell.alignment = Alignment(horizontal="left", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    
            # Coloration spécifique RGPD
            if c_idx == 5 and str(val).strip().lower() == "oui":
                cell.fill = fill_alert_red
                cell.font = Font(name="Segoe UI", size=10, bold=True, color="991B1B")
                
            # Coloration spécifique Criticité
            if c_idx == 4:
                crit_val = str(val).strip().lower()
                if crit_val == "haute":
                    cell.fill = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
                    cell.font = Font(name="Segoe UI", size=10, bold=True, color="92400E")
                elif crit_val == "moyenne":
                    cell.fill = PatternFill(start_color="EFF6FF", end_color="EFF6FF", fill_type="solid")
                    cell.font = Font(name="Segoe UI", size=10, bold=True, color="1E40AF")
                    
    # Validations listes de choix Excel
    dv_crit = DataValidation(type="list", formula1='"Faible,Moyenne,Haute"', allow_blank=True)
    dv_crit.prompt = 'Sélectionnez la criticité métier'
    dv_crit.promptTitle = 'Criticité'
    ws_func.add_data_validation(dv_crit)
    
    dv_rgpd = DataValidation(type="list", formula1='"Oui,Non"', allow_blank=True)
    dv_rgpd.prompt = 'Est-ce une donnée personnelle (RGPD) ?'
    dv_rgpd.promptTitle = 'RGPD'
    ws_func.add_data_validation(dv_rgpd)
    
    dv_source = DataValidation(type="list", formula1='"ERP,CRM,Excel manuel,API,Autre"', allow_blank=True)
    dv_source.prompt = 'Sélectionnez la source de la donnée'
    dv_source.promptTitle = 'Source de la donnée'
    ws_func.add_data_validation(dv_source)
    
    dv_freq = DataValidation(type="list", formula1='"Temps réel,Quotidienne,Hebdomadaire,Mensuelle,Ponctuelle"', allow_blank=True)
    dv_freq.prompt = 'Sélectionnez la fréquence de mise à jour'
    dv_freq.promptTitle = 'Fréquence'
    ws_func.add_data_validation(dv_freq)
    
    if nb_colonnes > 0:
        r_fin = nb_colonnes + 2
        dv_crit.add(f"D3:D{r_fin}")
        dv_rgpd.add(f"E3:E{r_fin}")
        dv_source.add(f"F3:F{r_fin}")
        dv_freq.add(f"G3:G{r_fin}")
        
    ws_func.row_dimensions[1].height = 25
    ws_func.row_dimensions[2].height = 30
    for r in range(3, nb_colonnes + 3):
        ws_func.row_dimensions[r].height = 22
    ws_func.auto_filter.ref = f"A2:J{nb_colonnes+2}"
    
    largeurs_func = {
        "A": 22, "B": 30, "C": 25, "D": 15, "E": 18, "F": 18, "G": 22, "H": 35, "I": 25, "J": 35
    }
    for col_lettre, l in largeurs_func.items():
        ws_func.column_dimensions[col_lettre].width = l
        
    # ============================================================
    #  FEUILLE 4 : LIGNAGE DES DONNÉES (optionnel)
    # ============================================================
    if lineage is not None and len(lineage.events()) > 0:
        generer_feuille_lignage(wb, lineage)

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


# ============================================================
#  5. GRAPHIQUES
# ============================================================

def graphique_jauge(score: float, chemin: str):
    """Crée une jauge circulaire (demi-cercle) du score global."""
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"aspect": "equal"})
    label, couleur, _ = niveau_qualite(score)

    # Demi-cercle de fond
    theta = np.linspace(np.pi, 0, 100)
    for i, (seuil, c) in enumerate([(60, "#dc2626"), (80, "#f59e0b"), (100, "#16a34a")]):
        debut = 0 if i == 0 else [60, 80][i - 1]
        t = np.linspace(np.pi * (1 - debut / 100), np.pi * (1 - seuil / 100), 50)
        ax.fill_between(np.cos(t), 0, np.sin(t), color=c, alpha=0.85)

    # Aiguille
    angle = np.pi * (1 - score / 100)
    ax.plot([0, 0.85 * np.cos(angle)], [0, 0.85 * np.sin(angle)],
            color="black", linewidth=3)
    ax.plot(0, 0, "o", color="black", markersize=10)

    ax.text(0, -0.25, f"{score}/100", ha="center", fontsize=22, fontweight="bold")
    ax.text(0, -0.45, label, ha="center", fontsize=14, color=couleur, fontweight="bold")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.2)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(chemin, dpi=120, bbox_inches="tight")
    plt.close(fig)


def graphique_qualite_par_colonne(completude: pd.DataFrame, coherence: pd.DataFrame,
                                  unicite: dict, chemin: str):
    """Barres horizontales : qualité par colonne, triée par criticité."""
    # Score simple par colonne : 100 - %manquantes - 10 par pb cohérence
    rows = []
    for col in completude["Nom de la colonne"]:
        pct = float(completude.loc[completude["Nom de la colonne"] == col,
                                   "% de données manquantes"].iloc[0])
        coh = coherence[coherence["Nom de la colonne"] == col].iloc[0]
        pen = 0
        pen += 15 if coh["Formats de dates incohérents"] == "Oui" else 0
        pen += 15 if coh["Valeurs négatives anormales"] > 0 else 0
        pen += 5 if coh["Valeurs aberrantes"] > 0 else 0
        pen += 5 if coh["Espaces superflus détectés"] > 0 else 0
        pen += 5 if coh["Majuscules/minuscules incohérentes"] == "Oui" else 0
        score = max(0, 100 - pct - pen)
        couleur = "#16a34a" if score >= 80 else ("#f59e0b" if score >= 60 else "#dc2626")
        rows.append((col, score, couleur))
    rows.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(rows))))
    cols = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    couleurs = [r[2] for r in rows]
    ax.barh(cols, scores, color=couleurs)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score qualité par colonne (/100)")
    ax.set_title("Qualité par colonne — triée du plus problématique au mieux noté")
    for i, v in enumerate(scores):
        ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(chemin, dpi=120, bbox_inches="tight")
    plt.close(fig)


def graphique_repartition_problemes(completude: pd.DataFrame, unicite: dict,
                                    coherence: pd.DataFrame, chemin: str):
    """Camembert : répartition des types de problèmes."""
    nb_manq = int((completude["Données manquantes"] > 0).sum())
    nb_doublons = unicite["doublons_lignes"] + int(
        (unicite["details"]["Problème d'unicité détecté"] == "Oui").sum())
    nb_incoh = int(
        (coherence["Formats de dates incohérents"] == "Oui").sum()
        + (coherence["Valeurs aberrantes"] > 0).sum()
        + (coherence["Espaces superflus détectés"] > 0).sum()
        + (coherence["Majuscules/minuscules incohérentes"] == "Oui").sum()
        + (coherence["Valeurs négatives anormales"] > 0).sum()
    )
    valeurs = [nb_manq, nb_doublons, nb_incoh]
    labels = ["Données manquantes", "Doublons / unicité", "Incohérences"]
    couleurs = ["#3b82f6", "#a855f7", "#f59e0b"]
    fig, ax = plt.subplots(figsize=(6, 4))
    if sum(valeurs) == 0:
        ax.text(0.5, 0.5, "Aucun problème détecté 🎉",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        ax.pie(valeurs, labels=labels, colors=couleurs, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 11})
        ax.set_title("Répartition des types de problèmes détectés")
    plt.tight_layout()
    plt.savefig(chemin, dpi=120, bbox_inches="tight")
    plt.close(fig)


def graphique_completude_barres(completude: pd.DataFrame, chemin: str):
    """Barres : % de complétude par colonne."""
    df = completude.copy()
    df["% complétude"] = 100 - df["% de données manquantes"]
    df = df.sort_values("% complétude")
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(df))))
    couleurs = ["#dc2626" if v < 90 else ("#f59e0b" if v < 100 else "#16a34a")
                for v in df["% complétude"]]
    ax.barh(df["Nom de la colonne"], df["% complétude"], color=couleurs)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de complétude")
    ax.set_title("Complétude par colonne")
    plt.tight_layout()
    plt.savefig(chemin, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
#  6. GÉNÉRATION DU RAPPORT PDF
# ============================================================

def generer_pdf(chemin_pdf: str, meta: dict, scores: dict,
                completude: pd.DataFrame, unicite: dict,
                coherence: pd.DataFrame, recos: list,
                lineage: DataLineage | None = None) -> str:
    """Génère un rapport PDF structuré (5 à 6 pages selon le lignage)."""

    tmp = tempfile.mkdtemp(prefix="dq_")
    p_jauge = os.path.join(tmp, "jauge.png")
    p_qualite_col = os.path.join(tmp, "qualite_col.png")
    p_camembert = os.path.join(tmp, "camembert.png")
    p_comp = os.path.join(tmp, "completude.png")

    graphique_jauge(scores["Score de qualité global"], p_jauge)
    graphique_qualite_par_colonne(completude, coherence, unicite, p_qualite_col)
    graphique_repartition_problemes(completude, unicite, coherence, p_camembert)
    graphique_completude_barres(completude, p_comp)

    label, couleur, emoji = niveau_qualite(scores["Score de qualité global"])

    doc = SimpleDocTemplate(chemin_pdf, pagesize=A4,
                            leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitreGarde", fontSize=28,
                              alignment=TA_CENTER, spaceAfter=20,
                              textColor=colors.HexColor("#0f172a")))
    styles.add(ParagraphStyle(name="SousTitre", fontSize=14,
                              alignment=TA_CENTER, spaceAfter=10,
                              textColor=colors.HexColor("#475569")))
    styles.add(ParagraphStyle(name="H1FR", fontSize=18, spaceAfter=12,
                              textColor=colors.HexColor("#0f172a")))
    styles.add(ParagraphStyle(name="H2FR", fontSize=13, spaceAfter=8,
                              textColor=colors.HexColor("#1e293b")))
    styles.add(ParagraphStyle(name="Corps", fontSize=10, spaceAfter=6,
                              textColor=colors.HexColor("#1f2937")))
    styles.add(ParagraphStyle(name="Reco", fontSize=10, spaceAfter=8,
                              leftIndent=10,
                              textColor=colors.HexColor("#1f2937")))

    story = []

    # --- PAGE 1 : Page de garde ---
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Rapport de Qualité des Données", styles["TitreGarde"]))
    story.append(Paragraph(f"Fichier analysé : <b>{meta['nom_fichier']}</b>",
                           styles["SousTitre"]))
    story.append(Paragraph(
        f"Date d'analyse : {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
        styles["SousTitre"]))
    story.append(Spacer(1, 1 * cm))
    story.append(RLImage(p_jauge, width=10 * cm, height=6 * cm))
    bandeau = Table([[f"  Qualité globale : {label}  "]],
                    colWidths=[16 * cm])
    bandeau.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(couleur)),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 16),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(bandeau)
    story.append(PageBreak())

    # --- PAGE 2 : Résumé exécutif ---
    story.append(Paragraph("Résumé Exécutif", styles["H1FR"]))
    pct_completude_globale = round(100 - completude["% de données manquantes"].mean(), 1)
    nb_critiques = sum(1 for r in recos if r["priorite"] == 0)
    chiffres = [
        ["Nombre de lignes", f"{meta['nb_lignes']:,}".replace(",", " ")],
        ["Nombre de colonnes", str(meta['nb_colonnes'])],
        ["% de complétude globale", f"{pct_completude_globale}%"],
        ["Problèmes critiques", str(nb_critiques)],
        ["Score de qualité global", f"{scores['Score de qualité global']}/100  ({label})"],
    ]
    t = Table(chiffres, colWidths=[8 * cm, 8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTSIZE", (0, 0), (-1, -1), 12),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("Les 3 problèmes prioritaires", styles["H2FR"]))
    top = top_3_problemes(recos)
    if top:
        for i, p in enumerate(top, 1):
            story.append(Paragraph(f"<b>{i}.</b> {_safe_pdf(p)}", styles["Corps"]))
    else:
        story.append(Paragraph("Aucun problème majeur détecté.", styles["Corps"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Action recommandée immédiate", styles["H2FR"]))
    if nb_critiques > 0:
        action = ("Traiter en priorité les problèmes critiques listés ci-dessus "
                  "avant toute exploitation des données.")
    elif scores["Score de qualité global"] >= 80:
        action = "Aucune action urgente. Le fichier peut être exploité en l'état."
    else:
        action = "Corriger les problèmes mineurs pour fiabiliser les analyses."
    story.append(Paragraph(action, styles["Corps"]))
    story.append(PageBreak())

    # --- PAGE 3 : Tableau récapitulatif par colonne ---
    story.append(Paragraph("Tableau récapitulatif par colonne", styles["H1FR"]))
    entetes = ["Niv.", "Colonne", "% manquantes", "Valeurs distinctes",
               "Anomalies cohérence"]
    data = [entetes]
    rouges, oranges, verts = [], [], []
    for _, row in completude.iterrows():
        col = row["Nom de la colonne"]
        niv = row["Niveau de criticité"]
        pct = row["% de données manquantes"]
        u = unicite["details"][unicite["details"]["Nom de la colonne"] == col].iloc[0]
        coh = coherence[coherence["Nom de la colonne"] == col].iloc[0]
        anomalies = []
        if coh["Formats de dates incohérents"] == "Oui": anomalies.append("dates mixtes")
        if coh["Valeurs négatives anormales"] > 0: anomalies.append("nég.")
        if coh["Valeurs aberrantes"] > 0: anomalies.append(f"{int(coh['Valeurs aberrantes'])} outliers")
        if coh["Espaces superflus détectés"] > 0: anomalies.append("espaces")
        if coh["Majuscules/minuscules incohérentes"] == "Oui": anomalies.append("casse")
        # Pire criticité parmi les 3 dimensions
        niv_global = niv
        if "🔴" in (coh["Niveau de criticité"], u["Niveau de criticité"], niv):
            niv_global = "🔴"
        elif niv_global == "🟢" and "🟡" in (coh["Niveau de criticité"], u["Niveau de criticité"]):
            niv_global = "🟡"
        data.append(["", col, f"{pct}%", str(u["Valeurs distinctes"]),
                     ", ".join(anomalies) if anomalies else "—"])
        idx = len(data) - 1
        (rouges if niv_global == "🔴" else
         oranges if niv_global == "🟡" else verts).append(idx)
    t = Table(data, colWidths=[1.2 * cm, 5 * cm, 2.5 * cm, 3 * cm, 5.3 * cm],
              repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    # Pastilles de couleur dans la colonne "Niv."
    for r in rouges:
        style.append(("BACKGROUND", (0, r), (0, r), colors.HexColor("#dc2626")))
        style.append(("BACKGROUND", (1, r), (-1, r), colors.HexColor("#fee2e2")))
    for r in oranges:
        style.append(("BACKGROUND", (0, r), (0, r), colors.HexColor("#f59e0b")))
        style.append(("BACKGROUND", (1, r), (-1, r), colors.HexColor("#fef3c7")))
    for r in verts:
        style.append(("BACKGROUND", (0, r), (0, r), colors.HexColor("#16a34a")))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<i>Légende : pastille verte = OK, orange = problème mineur, rouge = problème critique.</i>",
        styles["Corps"]))
    story.append(PageBreak())

    # --- PAGE 4 : Analyse détaillée ---
    story.append(Paragraph("Analyse détaillée par dimension", styles["H1FR"]))

    story.append(Paragraph("1. Complétude", styles["H2FR"]))
    story.append(Paragraph(
        "La complétude mesure si toutes les données attendues sont bien présentes. "
        "Une donnée manquante peut signifier un client incomplet, une transaction "
        "non tracée ou une erreur de saisie.", styles["Corps"]))
    story.append(RLImage(p_comp, width=16 * cm, height=8 * cm))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("2. Unicité", styles["H2FR"]))
    story.append(Paragraph(
        f"Nombre de lignes en doublon : <b>{unicite['doublons_lignes']}</b>. "
        "Les doublons faussent tous les comptages et indicateurs.",
        styles["Corps"]))

    story.append(Paragraph("3. Cohérence", styles["H2FR"]))
    anomalies_globales = sum(1 for _, r in coherence.iterrows()
                             if r["Niveau de criticité"] != "🟢")
    story.append(Paragraph(
        f"{anomalies_globales} colonne(s) présentent des anomalies de cohérence "
        "(formats, casse, valeurs aberrantes, espaces).", styles["Corps"]))
    story.append(RLImage(p_camembert, width=12 * cm, height=8 * cm))
    story.append(PageBreak())

    # --- PAGE 5 : Recommandations ---
    story.append(Paragraph("Recommandations détaillées", styles["H1FR"]))
    if not recos:
        story.append(Paragraph("Aucune recommandation : le fichier est de bonne qualité.",
                               styles["Corps"]))
    else:
        # Mapping niveau emoji -> badge texte coloré
        badge = {"🔴": '<font color="#dc2626"><b>[CRITIQUE]</b></font>',
                 "🟡": '<font color="#f59e0b"><b>[MINEUR]</b></font>',
                 "🟢": '<font color="#16a34a"><b>[OK]</b></font>'}
        for r in recos:
            b = badge.get(r["niveau"], "")
            bloc = (f"{b} <b>{r['titre']} — Colonne '{r['colonne']}'</b><br/>"
                    f"<b>Constat :</b> {_safe_pdf(r['constat'])}<br/>"
                    f"<b>Impact métier :</b> {_safe_pdf(r['impact'])}<br/>"
                    f"<b>Action recommandée :</b> {_safe_pdf(r['action'])}")
            story.append(Paragraph(bloc, styles["Reco"]))
            story.append(Spacer(1, 0.2 * cm))

    # --- PAGE 6 (optionnelle) : Lignage des données ---
    if lineage is not None and len(lineage.events()) > 0:
        story.append(PageBreak())
        story.append(Paragraph("Lignage des Données", styles["H1FR"]))
        story.append(Paragraph(
            "Historique des actions appliquees au jeu de donnees dans l'application.",
            styles["Corps"]))
        story.append(Spacer(1, 0.3 * cm))
        ligneage_data = [["Etape", "Action", "Description", "Horodatage"]]
        for e in lineage.events():
            ligneage_data.append([
                str(e.ordre), e.action, _safe_pdf(e.description), e.horodatage
            ])
        t_lin = Table(ligneage_data, colWidths=[1.5 * cm, 3 * cm, 8 * cm, 3.5 * cm],
                      repeatRows=1)
        t_lin.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_lin)

    doc.build(story)
    return chemin_pdf


# ============================================================
#  7. INTERFACE STREAMLIT
# ============================================================

def _charger_fichier_uploade(fichier_obj) -> pd.DataFrame | None:
    """Charge un fichier uploadé (UploadedFile) et retourne un DataFrame."""
    suffixe = os.path.splitext(fichier_obj.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffixe)
    tmp.write(fichier_obj.getbuffer())
    tmp.close()
    chemin = tmp.name
    try:
        return charger_fichier(chemin)
    except Exception:
        return None


def _format_nb(n: int) -> str:
    """Formate un nombre avec séparateurs de milliers (espaces)."""
    return f"{n:,}".replace(",", " ")


def lancer_streamlit():
    """Interface Streamlit complète avec multi-fichiers, merge et lignage."""
    import streamlit as st

    st.set_page_config(page_title="Data Quality Checker",
                       page_icon="📊", layout="wide")

    st.title("📊 Data Quality Checker")
    st.caption("Analyse de la qualité de vos données — pensée pour les profils métiers.")

    # ============================================================
    #  SECTION 1 : IMPORT MULTI-FICHIERS
    # ============================================================
    st.subheader("📥 Import des fichiers")
    fichiers = st.file_uploader(
        "Déposez un ou plusieurs fichiers CSV ou Excel",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    if not fichiers:
        st.info("👆 Importez au moins un fichier pour démarrer l'analyse.")
        return

    # Initialiser le session state pour les fichiers chargés
    if "loaded_files" not in st.session_state:
        st.session_state["loaded_files"] = {}

    # Charger chaque nouveau fichier
    for fich in fichiers:
        if fich.name not in st.session_state["loaded_files"]:
            df_loaded = _charger_fichier_uploade(fich)
            if df_loaded is not None:
                st.session_state["loaded_files"][fich.name] = df_loaded
            else:
                st.error(f"Impossible de charger le fichier : {fich.name}")

    if not st.session_state["loaded_files"]:
        st.error("Aucun fichier n'a pu être chargé.")
        return

    # Afficher les fichiers chargés
    noms_fichiers = list(st.session_state["loaded_files"].keys())
    st.success(f"✅ {len(noms_fichiers)} fichier(s) chargé(s) : {', '.join(noms_fichiers)}")

    # ============================================================
    #  SECTION 2 : CROISEMENT / FUSION (MERGE)
    # ============================================================
    if len(noms_fichiers) >= 2:
        with st.expander("🔗 Croisement / Fusion de deux fichiers", expanded=False):
            st.markdown(
                "Sélectionnez deux fichiers et une colonne clé commune pour les fusionner "
                "avant de lancer l'analyse de qualité."
            )
            col_g, col_d = st.columns(2)
            with col_g:
                fichier_primaire = st.selectbox("Fichier principal", noms_fichiers, key="merge_primary")
            with col_d:
                autres = [f for f in noms_fichiers if f != fichier_primaire]
                fichier_secondaire = st.selectbox("Fichier secondaire", autres, key="merge_secondary")

            df_primaire = st.session_state["loaded_files"][fichier_primaire]
            df_secondaire = st.session_state["loaded_files"][fichier_secondaire]

            colonnes_communes = find_common_columns(df_primaire, df_secondaire)
            if not colonnes_communes:
                st.warning("Aucune colonne commune détectée entre ces deux fichiers. La fusion est impossible.")
            else:
                cle_gauche = st.selectbox(
                    f"Colonne clé (dans {fichier_primaire})",
                    colonnes_communes, key="merge_key_left"
                )
                cle_droite = st.selectbox(
                    f"Colonne clé (dans {fichier_secondaire})",
                    colonnes_communes, key="merge_key_right"
                )
                join_type = st.selectbox(
                    "Type de jointure",
                    ["left", "inner", "outer"],
                    format_func={"left": "Left (garde tout le fichier principal)",
                                 "inner": "Inner (garde seulement les correspondances)",
                                 "outer": "Outer (garde tout)"},
                    key="merge_join_type"
                )
                if st.button("🔗 Fusionner les fichiers", use_container_width=True, key="btn_merge"):
                    try:
                        df_merged = merge_dataframes(
                            df_primaire, df_secondaire, cle_gauche, cle_droite,
                            how=join_type
                        )
                        nom_fusion = f"{fichier_primaire} + {fichier_secondaire}"
                        st.session_state["loaded_files"][nom_fusion] = df_merged
                        if "lineage" not in st.session_state:
                            st.session_state["lineage"] = DataLineage()
                        st.session_state["lineage"].add(
                            "merge",
                            f"Fusion de « {fichier_primaire} » et « {fichier_secondaire} » sur la clé « {cle_gauche} » (jointure {join_type})",
                            fichier_primaire=fichier_primaire,
                            fichier_secondaire=fichier_secondaire,
                            cle=cle_gauche,
                            join_type=join_type,
                        )
                        st.success(f"✅ Fusion réussie : {len(df_merged)} lignes, {len(df_merged.columns)} colonnes.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur lors de la fusion : {e}")

    # ============================================================
    #  SECTION 3 : SÉLECTION DU FICHIER À ANALYSER
    # ============================================================
    nom_fichier_analyse = st.selectbox(
        "📋 Fichier à analyser",
        list(st.session_state["loaded_files"].keys()),
        key="file_to_analyze"
    )
    df = st.session_state["loaded_files"][nom_fichier_analyse]

    if df is None or df.empty:
        st.error("Le fichier sélectionné est vide ou n'a pas pu être chargé.")
        return

    # Initialiser le lignage s'il n'existe pas
    if "lineage" not in st.session_state:
        st.session_state["lineage"] = DataLineage()
        st.session_state["lineage"].add(
            "import",
            f"Import du fichier « {nom_fichier_analyse} » ({len(df)} lignes, {len(df.columns)} colonnes)",
            nom_fichier=nom_fichier_analyse,
            nb_lignes=len(df),
            nb_colonnes=len(df.columns),
        )
    elif st.session_state.get("lineage_last_file") != nom_fichier_analyse:
        st.session_state["lineage"].add(
            "import",
            f"Sélection du fichier « {nom_fichier_analyse} » pour analyse ({len(df)} lignes, {len(df.columns)} colonnes)",
            nom_fichier=nom_fichier_analyse,
            nb_lignes=len(df),
            nb_colonnes=len(df.columns),
        )
    st.session_state["lineage_last_file"] = nom_fichier_analyse

    # ============================================================
    #  ANALYSE (avec cache pour la performance)
    # ============================================================
    @st.cache_data(show_spinner=False)
    def _analyse_cachee(_df_hash_key: str, _df: pd.DataFrame) -> dict:
        return lancer_analyse_complete(_df)

    # Utiliser une clé basée sur le nom du fichier + nombre de lignes
    cache_key = f"{nom_fichier_analyse}_{len(df)}_{len(df.columns)}"
    resultats = _analyse_cachee(cache_key, df)

    completude = resultats["completude"]
    unicite = resultats["unicite"]
    coherence = resultats["coherence"]
    scores = resultats["scores"]
    recos = resultats["recos"]
    label, couleur, emoji = niveau_qualite(scores["Score de qualité global"])
    nb_critiques = sum(1 for r in recos if r["priorite"] == 0)

    # Métadonnées
    mem = estimate_memory_usage(df)
    meta = {
        "nom_fichier": nom_fichier_analyse,
        "taille_ko": mem["ko"],
        "taille_mo": mem["mo"],
        "nb_lignes": int(df.shape[0]),
        "nb_colonnes": int(df.shape[1]),
        "colonnes": list(df.columns),
        "types": {c: str(df[c].dtype) for c in df.columns},
    }

    # ============================================================
    #  SIDEBAR
    # ============================================================
    with st.sidebar:
        st.header("🎯 Synthèse rapide")
        s_glob = scores["Score de qualité global"]
        s_comp = scores["Score de complétude"]
        s_uni = scores["Score d'unicité"]
        s_coh = scores["Score de cohérence"]
        st.metric("Score global", f"{s_glob}/100", label)
        st.metric("Complétude", f"{s_comp}/100")
        st.metric("Unicité", f"{s_uni}/100")
        st.metric("Cohérence", f"{s_coh}/100")
        st.divider()
        filtre_pb = st.checkbox("Afficher uniquement les colonnes avec problèmes",
                                value=False)
        st.divider()
        st.caption("Téléchargez le rapport complet :")
        pdf_path = os.path.join(tempfile.gettempdir(), "rapport_qualite.pdf")
        generer_pdf(pdf_path, meta, scores, completude, unicite, coherence, recos,
                    lineage=st.session_state.get("lineage"))
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Télécharger le rapport PDF", f.read(),
                               file_name=f"rapport_qualite_{os.path.splitext(nom_fichier_analyse)[0]}.pdf",
                               mime="application/pdf",
                               use_container_width=True)

    # ============================================================
    #  BANDEAU DE STATUT
    # ============================================================
    if scores["Score de qualité global"] >= 80:
        st.success(f"✅ Fichier de bonne qualité — Aucune action urgente requise (score {scores['Score de qualité global']}/100).")
    elif scores["Score de qualité global"] >= 60:
        st.warning(f"⚠️ Qualité moyenne — {nb_critiques} problème(s) critique(s) à corriger (score {scores['Score de qualité global']}/100).")
    else:
        st.error(f"🚨 Qualité critique — Intervention requise avant utilisation (score {scores['Score de qualité global']}/100).")

    # ============================================================
    #  RÉSUMÉ EXÉCUTIF
    # ============================================================
    st.subheader("📌 Résumé Exécutif")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(
            f"**Votre fichier `{nom_fichier_analyse}` contient "
            f"{_format_nb(meta['nb_lignes'])} lignes et {meta['nb_colonnes']} colonnes "
            f"(mémoire : {meta['taille_mo']} Mo).**  \n"
            f"La qualité globale est **:{('green' if label=='BONNE' else 'orange' if label=='MOYENNE' else 'red')}[{label}]** "
            f"avec un score de **{scores['Score de qualité global']}/100**."
        )
        st.markdown("**Top 3 des problèmes à traiter :**")
        top = top_3_problemes(recos)
        if top:
            for i, p in enumerate(top, 1):
                st.markdown(f"{i}. {p}")
        else:
            st.markdown("🎉 *Aucun problème majeur détecté.*")
        st.markdown("**Action recommandée immédiate :**")
        if nb_critiques > 0:
            st.markdown("➡️ Traiter les problèmes critiques avant d'utiliser ce fichier.")
        elif scores["Score de qualité global"] >= 80:
            st.markdown("➡️ Aucune action urgente : le fichier est exploitable.")
        else:
            st.markdown("➡️ Corriger les problèmes mineurs pour fiabiliser les analyses.")
    with col_b:
        p_jauge = os.path.join(tempfile.gettempdir(), "jauge_streamlit.png")
        graphique_jauge(scores["Score de qualité global"], p_jauge)
        st.image(p_jauge, use_container_width=True)

    st.divider()

    # ============================================================
    #  MÉTADONNÉES
    # ============================================================
    with st.expander("📁 Métadonnées du fichier", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fichier", nom_fichier_analyse)
        c2.metric("Mémoire", f"{meta['taille_mo']} Mo")
        c3.metric("Lignes", _format_nb(meta["nb_lignes"]))
        c4.metric("Colonnes", meta["nb_colonnes"])
        st.dataframe(df.head(10), use_container_width=True)

    # ============================================================
    #  ONGLETS
    # ============================================================
    onglets = st.tabs([
        "📋 Vue d'ensemble",
        "✅ Complétude",
        "🔁 Unicité",
        "🔍 Cohérence",
        "💡 Recommandations",
        "📊 Cartographie enrichie",
        "📁 Cartographie & Gouvernance",
        "🔗 Lignage des données",
    ])

    # --- Vue d'ensemble ---
    with onglets[0]:
        st.markdown("*Cette vue regroupe les indicateurs clés de qualité de votre fichier.*")
        col1, col2 = st.columns(2)
        with col1:
            p = os.path.join(tempfile.gettempdir(), "qualite_col.png")
            graphique_qualite_par_colonne(completude, coherence, unicite, p)
            st.image(p, caption="Qualité par colonne", use_container_width=True)
        with col2:
            p = os.path.join(tempfile.gettempdir(), "camembert.png")
            graphique_repartition_problemes(completude, unicite, coherence, p)
            st.image(p, caption="Répartition des problèmes", use_container_width=True)

    # --- Complétude ---
    with onglets[1]:
        st.markdown(
            "*La complétude mesure si toutes les données attendues sont bien présentes. "
            "Une donnée manquante peut signifier un client incomplet, "
            "une transaction non tracée ou une erreur de saisie.*"
        )
        df_aff = completude.copy()
        if filtre_pb:
            df_aff = df_aff[df_aff["Niveau de criticité"] != "🟢"]
        st.dataframe(df_aff, use_container_width=True, hide_index=True)
        p = os.path.join(tempfile.gettempdir(), "completude.png")
        graphique_completude_barres(completude, p)
        st.image(p, use_container_width=True)

    # --- Unicité ---
    with onglets[2]:
        st.markdown(
            "*L'unicité vérifie qu'il n'y a pas de doublons. "
            "Des doublons faussent les comptages, les calculs financiers "
            "et les indicateurs de pilotage.*"
        )
        st.metric("Lignes strictement identiques", unicite["doublons_lignes"])
        df_aff = unicite["details"].copy()
        if filtre_pb:
            df_aff = df_aff[df_aff["Niveau de criticité"] != "🟢"]
        st.dataframe(df_aff, use_container_width=True, hide_index=True)

    # --- Cohérence ---
    with onglets[3]:
        st.markdown(
            "*La cohérence vérifie que les valeurs respectent les règles attendues : "
            "formats homogènes, pas de valeurs aberrantes, casse uniforme, etc.*"
        )
        df_aff = coherence.copy()
        if filtre_pb:
            df_aff = df_aff[df_aff["Niveau de criticité"] != "🟢"]
        st.dataframe(df_aff, use_container_width=True, hide_index=True)

    # --- Recommandations ---
    with onglets[4]:
        st.markdown(
            "*Liste complète des actions recommandées, "
            "triées par priorité (critique en premier).*"
        )
        if not recos:
            st.success("🎉 Aucune recommandation : votre fichier est de bonne qualité.")
        else:
            for r in recos:
                if r["niveau"] == "🔴":
                    box = st.error
                elif r["niveau"] == "🟡":
                    box = st.warning
                else:
                    box = st.info
                box(
                    f"**{r['niveau']} {r['titre']} — Colonne '{r['colonne']}'**\n\n"
                    f"**Constat :** {r['constat']}\n\n"
                    f"**Impact métier :** {r['impact']}\n\n"
                    f"**Action recommandée :** {r['action']}"
                )

        if recos:
            csv_buf = io.StringIO()
            pd.DataFrame(recos).to_csv(csv_buf, index=False)
            st.download_button("⬇️ Exporter les recommandations (CSV)",
                               csv_buf.getvalue(),
                               file_name="recommandations.csv",
                               mime="text/csv")

    # --- Cartographie enrichie ---
    with onglets[5]:
        st.markdown(
            "### 📊 Cartographie enrichie des données\n"
            "Vue technique automatique : types détectés, cardinalité, "
            "échantillon de valeurs et taille mémoire par colonne."
        )
        df_carto_enrichie = enrichir_cartographie(df)
        st.dataframe(df_carto_enrichie, use_container_width=True, hide_index=True)

    # --- Cartographie & Gouvernance (dictionnaire fonctionnel) ---
    with onglets[6]:
        st.markdown(
            "### 📁 Cartographie et Gouvernance des Données\n"
            "Documentez le **dictionnaire des données** de votre fichier (métadonnées fonctionnelles) "
            "et générez un **livrable de cartographie complet** au format Excel."
        )

        if "df_carto" not in st.session_state or st.session_state.get("carto_file_name") != nom_fichier_analyse:
            st.session_state["df_carto"] = extraire_metadonnees_cartographie(df, completude, unicite, coherence)
            st.session_state["carto_file_name"] = nom_fichier_analyse

        with st.expander("📂 Importer/Recharger une cartographie existante (Excel)", expanded=False):
            st.markdown(
                "Importez votre fichier Excel de cartographie ci-dessous pour **réinjecter et restaurer automatiquement** "
                "toutes les métadonnées fonctionnelles saisies précédemment."
            )
            prev_excel = st.file_uploader("Choisissez un fichier de cartographie Excel (.xlsx)", type=["xlsx"], key="prev_excel_uploader")
            if prev_excel is not None:
                try:
                    df_prev_sheets = pd.read_excel(prev_excel, sheet_name=None)
                    df_merged = fusionner_cartographie_existante(st.session_state["df_carto"], df_prev_sheets)
                    st.session_state["df_carto"] = df_merged
                    st.success("✅ Importation réussie ! Les métadonnées fonctionnelles ont été rechargées.")
                except Exception as e:
                    st.error(f"Impossible de fusionner le fichier Excel importé : {e}")

        st.markdown("### 📊 Éditeur du Dictionnaire de Données")
        st.caption("💡 Double-cliquez sur les cellules pour documenter la partie fonctionnelle. Les colonnes techniques sont extraites automatiquement.")

        df_edite = st.data_editor(
            st.session_state["df_carto"],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Nom de la colonne": st.column_config.TextColumn(disabled=True),
                "Type technique": st.column_config.TextColumn(disabled=True),
                "Format détecté": st.column_config.TextColumn(disabled=True),
                "Complétude (%)": st.column_config.NumberColumn(disabled=True, format="%.1f%%"),
                "Valeurs distinctes": st.column_config.NumberColumn(disabled=True),
                "Longueur min": st.column_config.TextColumn(disabled=True),
                "Longueur max": st.column_config.TextColumn(disabled=True),
                "Valeur min": st.column_config.TextColumn(disabled=True),
                "Valeur max": st.column_config.TextColumn(disabled=True),
                "Valeur la plus fréquente": st.column_config.TextColumn(disabled=True),
                "Occurrences": st.column_config.NumberColumn(disabled=True),
                "Diagnostic Qualité": st.column_config.TextColumn(disabled=True),
                "Description Fonctionnelle": st.column_config.TextColumn(disabled=False, width="medium"),
                "Propriétaire de la donnée (Data Owner)": st.column_config.TextColumn(disabled=False),
                "Criticité Métier": st.column_config.SelectboxColumn(disabled=False, options=["Faible", "Moyenne", "Haute"]),
                "Sensibilité RGPD (Donnée Personnelle)": st.column_config.SelectboxColumn(disabled=False, options=["Non", "Oui"]),
                "Source de la donnée": st.column_config.SelectboxColumn(disabled=False, options=["ERP", "CRM", "Excel manuel", "API", "Autre"]),
                "Fréquence de mise à jour": st.column_config.SelectboxColumn(disabled=False, options=["Temps réel", "Quotidienne", "Hebdomadaire", "Mensuelle", "Ponctuelle"]),
                "Définition métier officielle": st.column_config.TextColumn(disabled=False, width="medium"),
                "Équipes consommatrices": st.column_config.TextColumn(disabled=False, width="medium"),
                "Lien avec d'autres données": st.column_config.TextColumn(disabled=False, width="medium")
            }
        )
        st.session_state["df_carto"] = df_edite

        st.markdown("---")
        st.markdown("### 📥 Télécharger le Livrable Officiel de Gouvernance")
        st.markdown(
            "Ce bouton génère un classeur Excel professionnel composé de **3 à 4 feuilles** :\n"
            "1. 🏠 **Accueil & Synthèse** : Dashboard de gouvernance.\n"
            "2. ⚙️ **Métadonnées Techniques** : Statistiques extraites automatiquement.\n"
            "3. 📁 **Dictionnaire Fonctionnel** : Champs métiers éditables avec listes déroulantes.\n"
            "4. 🔗 **Lignage des Données** : Historique des actions (si disponible)."
        )
        try:
            excel_data = generer_excel_cartographie(
                st.session_state["df_carto"], nom_fichier_analyse,
                lineage=st.session_state.get("lineage")
            )
            st.download_button(
                label="🟢 Télécharger la Cartographie & Dictionnaire (Excel)",
                data=excel_data,
                file_name=f"cartographie_{os.path.splitext(nom_fichier_analyse)[0]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Erreur lors de la génération de l'Excel : {e}")

    # --- Lignage des données ---
    with onglets[7]:
        st.markdown(
            "### 🔗 Lignage des Données\n"
            "Historique des actions appliquées au jeu de données dans l'application."
        )
        lineage = st.session_state.get("lineage")
        if lineage and len(lineage.events()) > 0:
            st.markdown("**Timeline :**")
            st.code(lineage.to_timeline_text(), language="text")
            st.markdown("---")
            st.dataframe(lineage.to_dataframe(), use_container_width=True, hide_index=True)
            st.markdown("---")
            csv_lineage = lineage.to_dataframe().to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Exporter le lignage (CSV)",
                csv_lineage,
                file_name=f"lignage_{os.path.splitext(nom_fichier_analyse)[0]}.csv",
                mime="text/csv",
            )
        else:
            st.info("Aucune action enregistrée pour le moment.")


# ============================================================
#  8. MODE CLI
# ============================================================

def lancer_cli(chemin: str):
    """Mode ligne de commande : analyse et génère un PDF."""
    print(f"\n📊 Analyse du fichier : {chemin}\n")
    df = charger_fichier(chemin)
    meta = obtenir_metadonnees(chemin, df)
    print(f"  → {meta['nb_lignes']} lignes, {meta['nb_colonnes']} colonnes "
          f"({meta['taille_ko']} Ko)")

    completude = analyser_completude(df)
    unicite = analyser_unicite(df)
    coherence = analyser_coherence(df)
    scores = calculer_scores(df, completude, unicite, coherence)
    recos = generer_recommandations(df, completude, unicite, coherence)

    label, _, _ = niveau_qualite(scores["Score de qualité global"])
    print(f"\n  🎯 Score global : {scores['Score de qualité global']}/100  ({label})")
    print(f"     • Complétude : {scores['Score de complétude']}/100")
    s_uni = scores["Score d'unicité"]
    print(f"     • Unicité    : {s_uni}/100")
    print(f"     • Cohérence  : {scores['Score de cohérence']}/100")

    print("\n  🔝 Top 3 des problèmes :")
    for i, p in enumerate(top_3_problemes(recos), 1):
        print(f"     {i}. {p}")

    base = os.path.splitext(os.path.basename(chemin))[0]
    sortie_pdf = f"rapport_qualite_{base}.pdf"
    generer_pdf(sortie_pdf, meta, scores, completude, unicite, coherence, recos)
    print(f"\n✅ Rapport PDF généré : {sortie_pdf}\n")


# ============================================================
#  9. POINT D'ENTRÉE
# ============================================================

def main():
    # Détection automatique : Streamlit ou CLI ?
    try:
        # Si streamlit est en train d'exécuter ce script, ce module est dispo
        import streamlit.runtime.scriptrunner as _sr  # noqa
        if _sr.get_script_run_ctx() is not None:
            lancer_streamlit()
            return
    except Exception:
        pass

    if len(sys.argv) < 2:
        print("Usage : python data_quality.py <fichier.csv|.xlsx>")
        print("   ou : streamlit run data_quality.py")
        sys.exit(1)
    try:
        lancer_cli(sys.argv[1])
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(2)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(3)
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")
        sys.exit(99)


if __name__ == "__main__":
    main()
