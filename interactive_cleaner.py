# -*- coding: utf-8 -*-
"""
Module de nettoyage interactif des données (Human-in-the-Loop).

Toutes les fonctions de ce module sont pures : elles ne dépendent
ni de Streamlit ni d'un chemin de fichier. Elles prennent un
DataFrame et une configuration en entrée et renvoient des résultats
exploitables par l'interface.

Le module expose :
  - CleanConfig : dataclass décrivant les règles de validation.
  - auto_detect_config : pré-remplit la config depuis les détections existantes.
  - detect_issues : détecte les anomalies par catégorie.
  - validate_edits : valide des corrections manuelles.
  - apply_edits : applique les corrections validées et journalise.
  - Correction automatiques sûres (strip, casse, dates, doublons).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd


# ============================================================
#  TYPES & CONSTANTES
# ============================================================

ColumnType = Literal[
    "texte", "nombre", "nombre_positif", "email", "telephone", "date", "code_postal"
]

PLACEHOLDERS: set[str] = {
    "n/a", "na", "null", "none", "x", "xx", "-", "?", "tbd", "inconnu", "inconnue",
    "nd", "n.d", "n.d.", "/", ".", "vide", "empty",
}

EMAIL_RE = re.compile(r"^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$")
TEL_RE = re.compile(r"^(?:\+33|0033|0)[1-9](?:[\s.-]?\d{2}){4}$")
CP_RE = re.compile(r"^\d{5}$")
DATE_PATTERNS: list[tuple[str, str]] = [
    ("%Y-%m-%d", r"^\d{4}-\d{2}-\d{2}$"),
    ("%d/%m/%Y", r"^\d{2}/\d{2}/\d{4}$"),
    ("%d-%m-%Y", r"^\d{2}-\d{2}-\d{4}$"),
    ("%Y/%m/%d", r"^\d{4}/\d{2}/\d{2}$"),
]


@dataclass
class CleanConfig:
    """Configuration des règles de validation pour le nettoyage."""
    colonnes_obligatoires: list[str] = field(default_factory=list)
    colonnes_uniques: list[str] = field(default_factory=list)
    types_colonnes: dict[str, ColumnType] = field(default_factory=dict)


# ============================================================
#  DÉTECTION AUTOMATIQUE DE CONFIGURATION
# ============================================================

def auto_detect_config(df: pd.DataFrame) -> CleanConfig:
    """Pré-remplit la config à partir des heuristiques déjà présentes dans l'outil."""
    obligatoires: list[str] = []
    uniques: list[str] = []
    types: dict[str, ColumnType] = {}

    for col in df.columns:
        nom = col.lower()
        s = df[col]

        # Colonnes obligatoires : nom suggère un identifiant ou contact clé
        if any(k in nom for k in ["id", "code", "email", "mail", "nom", "name", "prenom"]):
            obligatoires.append(col)

        # Colonnes uniques : nom suggère un identifiant
        if any(k in nom for k in ["id", "code", "email", "mail", "uuid", "ref"]):
            uniques.append(col)

        # Type attendu
        if pd.api.types.is_numeric_dtype(s):
            nom_lower = nom
            if any(k in nom_lower for k in ["prix", "price", "montant", "amount", "age", "quantite", "quantity", "stock", "ca", "chiffre", "salaire", "tarif"]):
                types[col] = "nombre_positif"
            else:
                types[col] = "nombre"
        elif pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "date"
        elif _detect_format_for_config(df[col]) == "Email":
            types[col] = "email"
        elif _detect_format_for_config(df[col]) == "Telephone":
            types[col] = "telephone"
        elif _detect_format_for_config(df[col]) == "Code postal":
            types[col] = "code_postal"
        elif _detect_format_for_config(df[col]) == "Date":
            types[col] = "date"
        else:
            types[col] = "texte"

    return CleanConfig(
        colonnes_obligatoires=obligatoires,
        colonnes_uniques=uniques,
        types_colonnes=types,
    )


def _detect_format_for_config(s: pd.Series) -> str:
    """Détecte le format dominant d'une colonne pour la config."""
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return "Texte"
    if pd.api.types.is_numeric_dtype(s):
        return "Numerique"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "Date"
    echantillon = s_clean.astype(str).str.strip().head(50)
    total = len(echantillon)
    if total == 0:
        return "Texte"

    nb_emails = sum(1 for v in echantillon if EMAIL_RE.match(v))
    nb_tels = sum(1 for v in echantillon if TEL_RE.match(v))
    nb_cps = sum(1 for v in echantillon if CP_RE.match(v))
    nb_dates = sum(1 for v in echantillon if any(p.search(v) for _, p in DATE_PATTERNS))

    seuil = max(1, total // 3)
    if nb_emails >= seuil:
        return "Email"
    if nb_tels >= seuil:
        return "Telephone"
    if nb_cps >= seuil:
        return "Code postal"
    if nb_dates >= seuil:
        return "Date"
    return "Texte"


# ============================================================
#  VALIDATION DE VALEURS
# ============================================================

def _is_missing_or_placeholder(val: Any, col: str, config: CleanConfig) -> bool:
    """True si la valeur est vide ou placeholder (sur colonne obligatoire)."""
    if pd.isna(val):
        return True
    s = str(val).strip()
    if s == "":
        return True
    if col in config.colonnes_obligatoires and s.lower() in PLACEHOLDERS:
        return True
    return False


def _validate_value(val: Any, col: str, config: CleanConfig) -> str | None:
    """Valide une valeur selon le type attendu. Retourne un message d'erreur ou None."""
    type_attendu = config.types_colonnes.get(col, "texte")

    # Vide / placeholder sur colonne obligatoire
    if _is_missing_or_placeholder(val, col, config):
        if col in config.colonnes_obligatoires:
            return f"Valeur vide ou non significative (« {str(val).strip()} ») refusée pour la colonne obligatoire « {col} »."
        return None  # Vide accepté sur colonne non obligatoire

    s = str(val).strip()

    if type_attendu == "email":
        if not EMAIL_RE.match(s):
            return f"« {s} » n'est pas un email valide pour la colonne « {col} »."
    elif type_attendu == "telephone":
        if not TEL_RE.match(s):
            return f"« {s} » n'est pas un numéro de téléphone valide pour la colonne « {col} »."
    elif type_attendu == "code_postal":
        if not CP_RE.match(s):
            return f"« {s} » n'est pas un code postal valide (5 chiffres) pour la colonne « {col} »."
    elif type_attendu == "date":
        if not _parse_date(s):
            return f"« {s} » n'est pas une date valide pour la colonne « {col} »."
    elif type_attendu in ("nombre", "nombre_positif"):
        try:
            num = float(s)
        except (ValueError, TypeError):
            return f"« {s} » n'est pas un nombre valide pour la colonne « {col} »."
        if type_attendu == "nombre_positif" and num < 0:
            return f"« {s} » est négatif alors que la colonne « {col} » doit contenir une valeur positive."

    return None


def _parse_date(s: str) -> datetime | None:
    """Tente de parser une date avec les formats connus."""
    for fmt, pat in DATE_PATTERNS:
        if re.match(pat, s.strip()):
            try:
                return datetime.strptime(s.strip(), fmt)
            except ValueError:
                pass
    # Essai large : pandas
    try:
        return pd.to_datetime(s, errors="raise").to_pydatetime()
    except Exception:
        return None


# ============================================================
#  DÉTECTION D'ANOMALIES
# ============================================================

@dataclass
class IssueRow:
    """Une anomalie détectée."""
    index: int        # index d'origine du DataFrame
    colonne: str
    valeur_actuelle: Any
    type_anomalie: str   # "manquante", "negatif", "email_invalide", etc.
    message: str


def detect_issues(
    df: pd.DataFrame, config: CleanConfig
) -> dict[str, list[IssueRow]]:
    """Détecte les anomalies par catégorie, indexées par l'index d'origine.

    Retourne un dict avec trois clés :
      - "manquantes" : cellules vides sur colonnes obligatoires
      - "aberrantes" : valeurs négatives anormales, formats invalides (email, tel, date, cp), espaces, casse
      - "doublons"   : doublons exacts et doublons après normalisation
    """
    manquantes: list[IssueRow] = []
    aberrantes: list[IssueRow] = []
    doublons: list[IssueRow] = []

    for col in df.columns:
        type_attendu = config.types_colonnes.get(col, "texte")
        s = df[col]

        for idx in df.index:
            val = s.loc[idx]

            # --- Manquantes sur colonnes obligatoires ---
            if col in config.colonnes_obligatoires and _is_missing_or_placeholder(val, col, config):
                manquantes.append(IssueRow(
                    index=idx, colonne=col, valeur_actuelle=val,
                    type_anomalie="manquante",
                    message=f"Ligne {idx}, colonne « {col} » : valeur manquante ou placeholder.",
                ))
                continue  # on ne double-compte pas comme aberrante

            if pd.isna(val) or str(val).strip() == "":
                continue  # vide sur colonne non obligatoire → pas une anomalie

            # --- Aberrantes / format ---
            s_val = str(val).strip() if not pd.isna(val) else ""

            if type_attendu == "nombre_positif" and pd.api.types.is_numeric_dtype(s):
                try:
                    num = float(val)
                    if num < 0:
                        aberrantes.append(IssueRow(
                            index=idx, colonne=col, valeur_actuelle=val,
                            type_anomalie="negatif",
                            message=f"Ligne {idx}, colonne « {col} » : valeur négative ({val}).",
                        ))
                except (ValueError, TypeError):
                    pass

            if type_attendu == "email" and not EMAIL_RE.match(s_val):
                aberrantes.append(IssueRow(
                    index=idx, colonne=col, valeur_actuelle=val,
                    type_anomalie="email_invalide",
                    message=f"Ligne {idx}, colonne « {col} » : email invalide (« {s_val} »).",
                ))

            if type_attendu == "telephone" and not TEL_RE.match(s_val):
                aberrantes.append(IssueRow(
                    index=idx, colonne=col, valeur_actuelle=val,
                    type_anomalie="tel_invalide",
                    message=f"Ligne {idx}, colonne « {col} » : téléphone invalide (« {s_val} »).",
                ))

            if type_attendu == "code_postal" and not CP_RE.match(s_val):
                aberrantes.append(IssueRow(
                    index=idx, colonne=col, valeur_actuelle=val,
                    type_anomalie="cp_invalide",
                    message=f"Ligne {idx}, colonne « {col} » : code postal invalide (« {s_val} »).",
                ))

            if type_attendu == "date" and not _parse_date(s_val):
                aberrantes.append(IssueRow(
                    index=idx, colonne=col, valeur_actuelle=val,
                    type_anomalie="date_invalide",
                    message=f"Ligne {idx}, colonne « {col} » : date invalide (« {s_val} »).",
                ))

            # Espaces superflus
            if s.dtype == object and not pd.isna(val):
                if str(val) != str(val).strip():
                    aberrantes.append(IssueRow(
                        index=idx, colonne=col, valeur_actuelle=val,
                        type_anomalie="espaces",
                        message=f"Ligne {idx}, colonne « {col} » : espaces superflus détectés.",
                    ))

    # --- Doublons ---
    doublons = _detect_duplicates(df, config)

    return {
        "manquantes": manquantes,
        "aberrantes": aberrantes,
        "doublons": doublons,
    }


def _detect_duplicates(df: pd.DataFrame, config: CleanConfig) -> list[IssueRow]:
    """Détecte les doublons exacts et après normalisation."""
    issues: list[IssueRow] = []

    # Doublons exacts : on garde la première occurrence, on signale les suivantes
    dup_mask = df.duplicated(keep="first")
    for idx in df.index[dup_mask]:
        issues.append(IssueRow(
            index=idx, colonne="(toutes colonnes)", valeur_actuelle="",
            type_anomalie="doublon_exact",
            message=f"Ligne {idx} : doublon exact d'une ligne précédente.",
        ))

    # Doublons après normalisation (strip + lower sur colonnes texte)
    df_norm = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df_norm[col] = df_norm[col].astype(str).str.strip().str.lower()
            df_norm[col] = df_norm[col].replace({"nan": np.nan})
    dup_norm_mask = df_norm.duplicated(keep="first")
    # On ne signale que ceux qui ne sont PAS déjà des doublons exacts
    for idx in df.index[dup_norm_mask & ~dup_mask]:
        issues.append(IssueRow(
            index=idx, colonne="(toutes colonnes)", valeur_actuelle="",
            type_anomalie="doublon_normalise",
            message=f"Ligne {idx} : doublon après normalisation (espaces/casse).",
        ))

    return issues


def detect_duplicate_clusters(
    df: pd.DataFrame, config: CleanConfig
) -> list[list[int]]:
    """Regroupe les doublons en clusters (listes d'index d'origine).

    Le premier élément de chaque cluster est l'occurrence de référence (conservée par défaut).
    """
    clusters: list[list[int]] = []
    seen: set[int] = set()

    df_norm = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df_norm[col] = df_norm[col].astype(str).str.strip().str.lower()
            df_norm[col] = df_norm[col].replace({"nan": np.nan})

    # Construire un hash par ligne normalisée
    for idx in df.index:
        if idx in seen:
            continue
        row_vals = df_norm.loc[idx].tolist()
        # Chercher toutes les lignes identiques (normalisées)
        mask = (df_norm == df_norm.loc[idx]).all(axis=1)
        cluster = list(df.index[mask])
        if len(cluster) > 1:
            clusters.append(cluster)
            seen.update(cluster)

    return clusters


# ============================================================
#  VALIDATION DES CORRECTIONS
# ============================================================

@dataclass
class Edit:
    """Une correction proposée par l'utilisateur."""
    index: int
    colonne: str
    nouvelle_valeur: Any


@dataclass
class ValidationResult:
    """Résultat de la validation d'un lot de corrections."""
    valides: list[Edit]
    erreurs: list[dict[str, str]]  # {"index": ..., "colonne": ..., "message": ...}


def validate_edits(
    edits: list[Edit], df: pd.DataFrame, config: CleanConfig
) -> ValidationResult:
    """Valide des corrections. Les invalides sont refusées individuellement."""
    valides: list[Edit] = []
    erreurs: list[dict[str, str]] = []

    for edit in edits:
        # Vérifier que la colonne existe
        if edit.colonne not in df.columns:
            erreurs.append({
                "index": str(edit.index),
                "colonne": edit.colonne,
                "message": f"La colonne « {edit.colonne } » n'existe pas dans le fichier.",
            })
            continue

        # Vérifier que l'index existe
        if edit.index not in df.index:
            erreurs.append({
                "index": str(edit.index),
                "colonne": edit.colonne,
                "message": f"La ligne {edit.index} n'existe plus dans le fichier.",
            })
            continue

        # Valider la valeur
        err = _validate_value(edit.nouvelle_valeur, edit.colonne, config)
        if err:
            erreurs.append({
                "index": str(edit.index),
                "colonne": edit.colonne,
                "message": err,
            })
        else:
            valides.append(edit)

    return ValidationResult(valides=valides, erreurs=erreurs)


# ============================================================
#  APPLICATION DES CORRECTIONS
# ============================================================

@dataclass
class ChangeEntry:
    """Une entrée du journal des modifications."""
    index: int
    colonne: str
    ancienne_valeur: Any
    nouvelle_valeur: Any
    type_correction: str   # "manuelle" ou "automatique"
    horodatage: str


def apply_edits(
    df: pd.DataFrame,
    edits: list[Edit],
    *,
    type_correction: str = "manuelle",
    change_log: list[ChangeEntry] | None = None,
) -> tuple[pd.DataFrame, list[ChangeEntry]]:
    """Applique les corrections validées et renvoie (nouveau_df, journal).

    Ne modifie PAS le DataFrame d'origine. Chaque correction ne touche qu'une
    cellule précise (index, colonne), jamais une ligne entière.
    """
    df_out = df.copy()
    log = change_log if change_log is not None else []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for edit in edits:
        old_val = df_out.at[edit.index, edit.colonne]
        df_out.at[edit.index, edit.colonne] = edit.nouvelle_valeur
        log.append(ChangeEntry(
            index=edit.index,
            colonne=edit.colonne,
            ancienne_valeur=old_val,
            nouvelle_valeur=edit.nouvelle_valeur,
            type_correction=type_correction,
            horodatage=now,
        ))

    return df_out, log


def apply_row_deletions(
    df: pd.DataFrame,
    indices: list[int],
    *,
    change_log: list[ChangeEntry] | None = None,
) -> tuple[pd.DataFrame, list[ChangeEntry]]:
    """Supprime les lignes par index et journalise."""
    df_out = df.copy()
    log = change_log if change_log is not None else []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Garder seulement les indices qui existent
    indices_valides = [i for i in indices if i in df_out.index]
    if not indices_valides:
        return df_out, log

    for idx in indices_valides:
        log.append(ChangeEntry(
            index=idx,
            colonne="(ligne supprimée)",
            ancienne_valeur="",
            nouvelle_valeur="",
            type_correction="automatique",
            horodatage=now,
        ))

    df_out = df_out.drop(index=indices_valides)
    return df_out, log


# ============================================================
#  CORRECTIONS AUTOMATIQUES SÛRES
# ============================================================

def auto_strip_spaces(
    df: pd.DataFrame, columns: list[str] | None = None,
    *, change_log: list[ChangeEntry] | None = None,
) -> tuple[pd.DataFrame, list[ChangeEntry], int]:
    """Supprime les espaces en début/fin de toutes les cellules texte.

    Retourne (nouveau_df, journal, nb_cellules_modifiees).
    """
    df_out = df.copy()
    log = change_log if change_log is not None else []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = columns if columns is not None else list(df.columns)
    count = 0

    for col in cols:
        if col not in df_out.columns:
            continue
        if df_out[col].dtype != object:
            continue
        for idx in df_out.index:
            val = df_out.at[idx, col]
            if pd.isna(val):
                continue
            s = str(val)
            stripped = s.strip()
            if stripped != s:
                old = df_out.at[idx, col]
                df_out.at[idx, col] = stripped
                log.append(ChangeEntry(
                    index=idx, colonne=col,
                    ancienne_valeur=old, nouvelle_valeur=stripped,
                    type_correction="automatique", horodatage=now,
                ))
                count += 1

    return df_out, log, count


def auto_standardize_case(
    df: pd.DataFrame, col: str, mode: Literal["lower", "upper", "title"],
    *, change_log: list[ChangeEntry] | None = None,
) -> tuple[pd.DataFrame, list[ChangeEntry], int]:
    """Uniformise la casse d'une colonne.

    mode : "lower" (minuscules), "upper" (MAJUSCULES), "title" (Nom Propre).
    """
    df_out = df.copy()
    log = change_log if change_log is not None else []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    count = 0

    if col not in df_out.columns or df_out[col].dtype != object:
        return df_out, log, 0

    for idx in df_out.index:
        val = df_out.at[idx, col]
        if pd.isna(val):
            continue
        s = str(val).strip()
        if mode == "lower":
            new_val = s.lower()
        elif mode == "upper":
            new_val = s.upper()
        else:
            new_val = s.title()
        if new_val != str(val):
            old = df_out.at[idx, col]
            df_out.at[idx, col] = new_val
            log.append(ChangeEntry(
                index=idx, colonne=col,
                ancienne_valeur=old, nouvelle_valeur=new_val,
                type_correction="automatique", horodatage=now,
            ))
            count += 1

    return df_out, log, count


def auto_standardize_dates(
    df: pd.DataFrame, col: str, target_format: str = "%Y-%m-%d",
    *, change_log: list[ChangeEntry] | None = None,
) -> tuple[pd.DataFrame, list[ChangeEntry], int]:
    """Standardise les dates d'une colonne vers un format cible.

    Parse chaque valeur avec les formats connus, puis reformate vers target_format.
    Les valeurs non parsables sont laissées inchangées.
    """
    df_out = df.copy()
    log = change_log if change_log is not None else []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    count = 0

    if col not in df_out.columns:
        return df_out, log, 0

    for idx in df_out.index:
        val = df_out.at[idx, col]
        if pd.isna(val):
            continue
        s = str(val).strip()
        parsed = _parse_date(s)
        if parsed is None:
            continue
        new_val = parsed.strftime(target_format)
        if new_val != s:
            old = df_out.at[idx, col]
            df_out.at[idx, col] = new_val
            log.append(ChangeEntry(
                index=idx, colonne=col,
                ancienne_valeur=old, nouvelle_valeur=new_val,
                type_correction="automatique", horodatage=now,
            ))
            count += 1

    return df_out, log, count


def auto_remove_exact_duplicates(
    df: pd.DataFrame,
    *, change_log: list[ChangeEntry] | None = None,
) -> tuple[pd.DataFrame, list[ChangeEntry], int]:
    """Supprime les doublons exacts (garde la première occurrence).

    Retourne (nouveau_df, journal, nb_lignes_supprimees).
    """
    log = change_log if change_log is not None else []
    dup_mask = df.duplicated(keep="first")
    indices = list(df.index[dup_mask])
    df_out, log = apply_row_deletions(df, indices, change_log=log)
    return df_out, log, len(indices)


# ============================================================
#  EXPORT DU JOURNAL
# ============================================================

def change_log_to_dataframe(change_log: list[ChangeEntry]) -> pd.DataFrame:
    """Convertit le journal des modifications en DataFrame pour affichage/export."""
    if not change_log:
        return pd.DataFrame(columns=[
            "Index", "Colonne", "Ancienne valeur", "Nouvelle valeur",
            "Type de correction", "Horodatage",
        ])
    return pd.DataFrame([
        {
            "Index": e.index,
            "Colonne": e.colonne,
            "Ancienne valeur": e.ancienne_valeur,
            "Nouvelle valeur": e.nouvelle_valeur,
            "Type de correction": e.type_correction,
            "Horodatage": e.horodatage,
        }
        for e in change_log
    ])


# ============================================================
#  UTILITAIRES POUR L'INTERFACE
# ============================================================

def issues_to_dataframe(issues: list[IssueRow]) -> pd.DataFrame:
    """Convertit une liste d'anomalies en DataFrame pour affichage."""
    if not issues:
        return pd.DataFrame(columns=["Index", "Colonne", "Valeur actuelle", "Type", "Message"])
    return pd.DataFrame([
        {
            "Index": i.index,
            "Colonne": i.colonne,
            "Valeur actuelle": i.valeur_actuelle,
            "Type": i.type_anomalie,
            "Message": i.message,
        }
        for i in issues
    ])


def extract_edits_from_editor(
    edited_rows: dict[int, dict[str, Any]],
    original_df: pd.DataFrame,
) -> list[Edit]:
    """Extrait les corrections depuis le résultat edited_rows de st.data_editor.

    st.data_editor renvoie un dict {row_position: {col_name: new_value}}
    où row_position est la position dans le DataFrame affiché (pas l'index d'origine).
    On utilise original_df pour mapper vers l'index d'origine.
    """
    edits: list[Edit] = []
    for row_pos, changes in edited_rows.items():
        if row_pos < 0 or row_pos >= len(original_df):
            continue
        original_index = original_df.index[row_pos]
        for col_name, new_val in changes.items():
            edits.append(Edit(
                index=original_index,
                colonne=col_name,
                nouvelle_valeur=new_val,
            ))
    return edits
