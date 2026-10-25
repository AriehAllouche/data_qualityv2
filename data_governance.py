# -*- coding: utf-8 -*-
"""
Module de gouvernance des données : lignage, fusion multi-fichiers
et enrichissement de la cartographie.

Toutes les fonctions sont pures (pas de Streamlit, pas de chemin de fichier).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd


# ============================================================
#  LIGNAGE DES DONNÉES
# ============================================================

@dataclass
class LineageEvent:
    """Une étape du lignage des données."""
    ordre: int
    action: str          # "import", "merge", "nettoyage", "export", etc.
    description: str     # description lisible
    horodatage: str
    details: dict[str, Any] = field(default_factory=dict)


class DataLineage:
    """Trace l'historique des actions subies par un jeu de données."""

    def __init__(self) -> None:
        self._events: list[LineageEvent] = []
        self._counter: int = 0

    def add(self, action: str, description: str, **details: Any) -> LineageEvent:
        """Ajoute un événement au lignage."""
        self._counter += 1
        event = LineageEvent(
            ordre=self._counter,
            action=action,
            description=description,
            horodatage=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            details=details,
        )
        self._events.append(event)
        return event

    def events(self) -> list[LineageEvent]:
        return list(self._events)

    def to_dataframe(self) -> pd.DataFrame:
        """Convertit le lignage en DataFrame pour affichage/export."""
        if not self._events:
            return pd.DataFrame(columns=["Étape", "Action", "Description", "Horodatage", "Détails"])
        rows = []
        for e in self._events:
            details_str = ", ".join(f"{k}={v}" for k, v in e.details.items()) if e.details else ""
            rows.append({
                "Étape": e.ordre,
                "Action": e.action,
                "Description": e.description,
                "Horodatage": e.horodatage,
                "Détails": details_str,
            })
        return pd.DataFrame(rows)

    def to_timeline_text(self) -> str:
        """Renvoie le lignage sous forme de timeline texte."""
        if not self._events:
            return "Aucune action enregistrée."
        lines: list[str] = []
        for e in self._events:
            icon = _ICONS.get(e.action, "•")
            lines.append(f"{icon} Étape {e.ordre} — {e.description}  ({e.horodatage})")
        return "\n".join(lines)


_ICONS: dict[str, str] = {
    "import": "📥",
    "merge": "🔗",
    "nettoyage": "🧹",
    "export": "📤",
    "correction_manuelle": "✏️",
    "correction_auto": "⚙️",
    "suppression": "🗑️",
}


# ============================================================
#  FUSION MULTI-FICHIERS
# ============================================================

JoinType = Literal["inner", "left", "outer"]


def merge_dataframes(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_key: str,
    right_key: str,
    how: JoinType = "left",
    suffixes: tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    """Fusionne deux DataFrames sur une colonne clé commune.

    Retourne le DataFrame fusionné. Les colonnes en conflit reçoivent un suffixe.
    """
    if left_key not in df_left.columns:
        raise ValueError(f"La colonne « {left_key} » n'existe pas dans le premier fichier.")
    if right_key not in df_right.columns:
        raise ValueError(f"La colonne « {right_key} » n'existe pas dans le second fichier.")

    df_merged = df_left.merge(
        df_right,
        left_on=left_key,
        right_on=right_key,
        how=how,
        suffixes=suffixes,
    )
    return df_merged


def find_common_columns(
    df_left: pd.DataFrame, df_right: pd.DataFrame
) -> list[str]:
    """Retourne les colonnes communes aux deux DataFrames."""
    return list(set(df_left.columns) & set(df_right.columns))


# ============================================================
#  ENRICHISSEMENT DE LA CARTOGRAPHIE
# ============================================================

def estimate_memory_usage(df: pd.DataFrame) -> dict[str, float]:
    """Estime la taille en mémoire du DataFrame.

    Retourne un dict avec 'ko' et 'mo'.
    """
    bytes_total = df.memory_usage(deep=True).sum()
    return {
        "ko": round(bytes_total / 1024, 2),
        "mo": round(bytes_total / (1024 * 1024), 3),
    }


def enrichir_cartographie(df: pd.DataFrame) -> pd.DataFrame:
    """Génère une cartographie enrichie avec types détectés, cardinalité,
    échantillon de 5 valeurs et taille mémoire par colonne."""
    rows: list[dict[str, Any]] = []
    mem_total = estimate_memory_usage(df)

    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)

        # Type technique lisible
        if "int" in dtype_str:
            type_tech = "Integer"
        elif "float" in dtype_str:
            type_tech = "Float"
        elif "bool" in dtype_str:
            type_tech = "Boolean"
        elif "datetime" in dtype_str:
            type_tech = "Date"
        elif dtype_str == "object":
            type_tech = "String"
        else:
            type_tech = dtype_str

        # Cardinalité
        cardinalite = int(s.nunique(dropna=True))

        # Échantillon des 5 premières valeurs non-nulles
        echantillon = s.dropna().head(5).tolist()
        echantillon_str = " | ".join(str(v) for v in echantillon)

        # Taille mémoire par colonne
        mem_col = s.memory_usage(deep=True, index=False)
        mem_col_ko = round(mem_col / 1024, 2)

        # Type sémantique détecté (réutilise les heuristiques)
        type_semantique = _detect_semantic_type(s)

        rows.append({
            "Nom de la colonne": col,
            "Type technique": type_tech,
            "Type sémantique": type_semantique,
            "Cardinalité": cardinalite,
            "Échantillon (5 valeurs)": echantillon_str,
            "Mémoire (Ko)": mem_col_ko,
            "% manquantes": round(100 * s.isna().sum() / max(1, len(s)), 2),
            "Mémoire totale (Mo)": mem_total["mo"],
        })

    return pd.DataFrame(rows)


def _detect_semantic_type(s: pd.Series) -> str:
    """Détecte le type sémantique d'une colonne (Email, Téléphone, Date, Code postal, Texte)."""
    import re as _re

    s_clean = s.dropna()
    if len(s_clean) == 0:
        return "Texte"
    if pd.api.types.is_numeric_dtype(s):
        return "Numérique"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "Date"

    ech = s_clean.astype(str).str.strip().head(50)
    total = len(ech)
    if total == 0:
        return "Texte"

    email_pat = _re.compile(r"^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$")
    tel_pat = _re.compile(r"^(?:\+33|0033|0)[1-9](?:[\s.-]?\d{2}){4}$")
    cp_pat = _re.compile(r"^\d{5}$")
    date_pat = _re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$")

    nb_emails = sum(1 for v in ech if email_pat.match(v))
    nb_tels = sum(1 for v in ech if tel_pat.match(v))
    nb_cps = sum(1 for v in ech if cp_pat.match(v))
    nb_dates = sum(1 for v in ech if date_pat.match(v))

    seuil = max(1, total // 3)
    if nb_emails >= seuil:
        return "Email"
    if nb_tels >= seuil:
        return "Téléphone"
    if nb_cps >= seuil:
        return "Code postal"
    if nb_dates >= seuil:
        return "Date"
    return "Texte"


# ============================================================
#  EXPORT EXCEL : FEUILLE LIGNAGE
# ============================================================

def generer_feuille_lignage(wb, lineage: DataLineage) -> None:
    """Ajoute une feuille « Lignage des données » à un workbook openpyxl.

    wb doit être un workbook openpyxl déjà créé.
    """
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    fill_header = PatternFill(start_color="1E40AF", end_color="1E40AF", fill_type="solid")
    fill_zebra_light = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    fill_zebra_white = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
    font_header = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    font_normal = Font(name="Segoe UI", size=10, color="0F172A")
    border_thin = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    ws = wb.create_sheet(title="Lignage des Données")
    ws.views.sheetView[0].showGridLines = True

    ws.merge_cells("A1:E1")
    ws["A1"] = "LIGNAGE DES DONNÉES — HISTORIQUE DES ACTIONS"
    ws["A1"].font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    ws["A1"].fill = fill_header
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")

    headers = ["Étape", "Action", "Description", "Horodatage", "Détails"]
    for c_idx, h in enumerate(headers, 1):
        cell = ws.cell(row=2, column=c_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border_thin

    df_lineage = lineage.to_dataframe()
    if len(df_lineage) > 0:
        for r_idx, row_data in enumerate(df_lineage.itertuples(index=False), 3):
            zebra = fill_zebra_light if r_idx % 2 == 0 else fill_zebra_white
            for c_idx, val in enumerate(row_data, 1):
                cell = ws.cell(row=r_idx, column=c_idx)
                cell.value = val
                cell.font = font_normal
                cell.fill = zebra
                cell.border = border_thin
                if c_idx == 1:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")
    else:
        ws.cell(row=3, column=1, value="Aucune action enregistrée").font = font_normal

    ws.row_dimensions[1].height = 25
    ws.row_dimensions[2].height = 25

    largeurs = {"A": 10, "B": 20, "C": 50, "D": 20, "E": 40}
    for lettre, largeur in largeurs.items():
        ws.column_dimensions[lettre].width = largeur


# ============================================================
#  EXPORT CSV / EXCEL DU FICHIER NETTOYÉ
# ============================================================

def export_clean_csv(
    df: pd.DataFrame, separator: str = ";", encoding: str = "utf-8-sig"
) -> bytes:
    """Exporte un DataFrame en CSV bytes avec l'encodage et le séparateur choisis."""
    buf = pd.io.common.StringIO()
    df.to_csv(buf, sep=separator, encoding=encoding, index=False)
    return buf.getvalue().encode(encoding)


def export_clean_excel(df: pd.DataFrame) -> bytes:
    """Exporte un DataFrame en Excel bytes."""
    import io
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()
