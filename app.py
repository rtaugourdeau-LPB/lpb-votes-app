import re
import time
from typing import List, Tuple, Optional
from datetime import datetime

import streamlit as st
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from rapidfuzz import process, fuzz
from unidecode import unidecode

# =======================
# 🔐 CONFIG EN BRUT
# =======================
import os

AIRTABLE_TOKEN   = st.secrets["AIRTABLE_TOKEN"]
AIRTABLE_BASE_ID = st.secrets["AIRTABLE_BASE_ID"]

PG_HOST     = st.secrets["PG_HOST"]
PG_PORT     = st.secrets["PG_PORT"]
PG_DB       = st.secrets["PG_DB"]
PG_USER     = st.secrets["PG_USER"]
PG_PASSWORD = st.secrets["PG_PASSWORD"]
PG_SSLMODE  = st.secrets.get("PG_SSLMODE", "require")

AIR_H = {"Authorization": f"Bearer {AIRTABLE_TOKEN}"} if AIRTABLE_TOKEN else None

# URL publique projet LPB
LPB_PROJECT_URL = "https://app.lapremierebrique.fr/fr/projects/{project_id}"

# =================================================================================
# app.py — LPB — Croisement Votes Airtable ↔ Souscriptions BO
#   pip install streamlit pandas sqlalchemy psycopg2-binary requests rapidfuzz unidecode
#   streamlit run app.py
# =================================================================================

# =======================
# Utils HTTP / strings
# =======================

def ensure_ok(r: requests.Response):
    try:
        r.raise_for_status()
    except Exception as e:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"HTTP {r.status_code}: {detail}") from e


def normalize_name(s: str) -> str:
    """Normalisation douce pour comparer les noms."""
    if s is None:
        return ""
    s2 = unidecode(str(s)).lower().strip()
    s2 = re.sub(r"\s+", " ", s2)
    s2 = re.sub(r"^(le|la|les|l')\s+", "", s2)
    return s2


def normalize_email(s: str) -> str:
    if s is None:
        return ""
    return unidecode(str(s)).strip().lower()


def extract_dept(token: str) -> Optional[int]:
    """Extrait (DD) ou (DDD) en fin de libellé → int, sinon None."""
    m = re.search(r"\((\d{2,3})\)\s*$", str(token).strip())
    return int(m.group(1)) if m else None


def make_nom_airtable(name, department) -> str:
    """Construit 'Nom (DD)' à partir de BO (name, department)."""
    n = "" if pd.isna(name) else str(name).strip()
    if pd.isna(department) or str(department).strip() == "":
        return n
    try:
        dep = int(department)
    except Exception:
        dep = str(department).strip()
    return f"{n} ({dep})"

# =======================
# Patterns globaux prolongation / pouvoir
# =======================

PROLONG_PATTERN = re.compile(
    r"(prolongation|pr[ée]pa|pr[ée]par|pr[ée]paration|prepa|ne\s*se\s*pron|nspp)",
    re.I,
)
POUVOIR_PATTERN = re.compile(
    r"(pouvoir|procuration|proxy)",
    re.I,
)

# =======================
# Airtable API helpers
# =======================

def list_tables_with_views(base_id: str) -> List[dict]:
    """Retourne la métadonnée Airtable de toutes les tables de la base."""
    if not AIR_H:
        return []
    r = requests.get(
        f"https://api.airtable.com/v0/meta/bases/{base_id}/tables",
        headers=AIR_H,
        timeout=30,
    )
    ensure_ok(r)
    return r.json().get("tables", []) or []


def table_has_prolongation_or_pouvoir(table_meta: dict) -> bool:
    """
    Ne garde que les tables qui ont AU MOINS une colonne liée à
    'prolongation' OU 'pouvoir'.
    """
    for f in table_meta.get("fields", []) or []:
        name = str(f.get("name", ""))
        if PROLONG_PATTERN.search(name) or POUVOIR_PATTERN.search(name):
            return True
    return False


def fetch_view_records(base_id: str, table_id_or_name: str, view_id_or_name: str, page_size: int = 100) -> List[dict]:
    if not AIR_H:
        return []
    url = f"https://api.airtable.com/v0/{base_id}/{table_id_or_name}"
    params = {"pageSize": page_size, "view": view_id_or_name}
    out, offset = [], None
    while True:
        if offset:
            params["offset"] = offset
        r = requests.get(url, headers=AIR_H, params=params, timeout=60)
        ensure_ok(r)
        data = r.json()
        out.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break
        time.sleep(0.12)  # respect soft-rate
    return out


def flatten(records: List[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = []
    for r in records:
        d = {"_air_id": r.get("id"), "_air_createdTime": r.get("createdTime")}
        d.update(r.get("fields", {}) or {})
        rows.append(d)
    return pd.DataFrame(rows)

# =======================
# BO engine & chargement
# =======================

@st.cache_resource(show_spinner=False)
def get_engine():
    if not all([PG_HOST, PG_DB, PG_USER, PG_PASSWORD]):
        return None
    uri = (
        f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
        f"@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode={PG_SSLMODE}"
    )
    return create_engine(uri, pool_pre_ping=True)


@st.cache_data(show_spinner=True)
def load_projects_df() -> pd.DataFrame:
    eng = get_engine()
    if eng is None:
        return pd.DataFrame(columns=["id", "name", "department", "nom_airtable", "name_norm", "nom_airtable_norm"])
    q = text(
        """
        SELECT id, name, department
        FROM public.projects
        WHERE name IS NOT NULL
        """
    )
    df = pd.read_sql(q, eng)
    df["nom_airtable"] = df.apply(lambda r: make_nom_airtable(r["name"], r["department"]), axis=1)
    df["name_norm"] = df["name"].apply(normalize_name)
    df["nom_airtable_norm"] = df["nom_airtable"].apply(normalize_name)
    return df


@st.cache_data(show_spinner=True)
def load_subs_for_project(project_id: int) -> pd.DataFrame:
    eng = get_engine()
    if eng is None:
        return pd.DataFrame(
            columns=["subscription_id", "users_profile_id", "project_id", "subscribed_at", "email_normalized", "email_raw"]
        )
    q = text(
        """
        SELECT
            s.id AS subscription_id,
            s.users_profile_id AS users_profile_id,
            s.project_id,
            s.created_at AS subscribed_at,
            lower(trim(u.email)) AS email_normalized,
            u.email AS email_raw
        FROM public.subscriptions s
        JOIN public.users_profiles up ON up.id = s.users_profile_id
        LEFT JOIN public.users u ON u.id = up.user_id
        WHERE s.status <> 'canceled'
          AND s.project_id = :pid
          AND u.email IS NOT NULL
        """
    )
    return pd.read_sql(q, eng, params={"pid": int(project_id)})

# =======================
# Matching Projet (Airtable → BO)
# =======================

def guess_candidates_from_label(label: str, projects_df: pd.DataFrame, topn: int = 10) -> List[Tuple[int, str, float]]:
    cand: List[Tuple[int, str, float]] = []
    label = (label or "").strip()
    dept = extract_dept(label)
    label_no_dep = re.sub(r"\s*\(\d{2,3}\)\s*$", "", label).strip()

    # 1) exact name + dept
    if dept is not None:
        m1 = projects_df.query("department == @dept and name == @label_no_dep")
        for _, r in m1.iterrows():
            cand.append((int(r["id"]), r["nom_airtable"], 100.0))

    # 2) exact name seul
    m2 = projects_df.query("name == @label_no_dep")
    for _, r in m2.iterrows():
        tup = (int(r["id"]), r["nom_airtable"], 99.0)
        if tup not in cand:
            cand.append(tup)

    # 3) exact normalisé
    label_norm = normalize_name(label_no_dep)
    m3 = projects_df[projects_df["name_norm"] == label_norm]
    for _, r in m3.iterrows():
        tup = (int(r["id"]), r["nom_airtable"], 98.0)
        if tup not in cand:
            cand.append(tup)

    # 4) fuzzy sur nom_airtable puis name
    choices = projects_df["nom_airtable"].tolist()
    for (disp, score, pos) in process.extract(label, choices, scorer=fuzz.token_set_ratio, limit=topn):
        r = projects_df.iloc[pos]
        tup = (int(r["id"]), r["nom_airtable"], float(score))
        if tup not in cand:
            cand.append(tup)

    choices2 = projects_df["name"].tolist()
    for (disp, score, pos) in process.extract(label_no_dep, choices2, scorer=fuzz.token_set_ratio, limit=topn):
        r = projects_df.iloc[pos]
        tup = (int(r["id"]), r["nom_airtable"], float(score) - 0.5)
        if tup not in cand:
            cand.append(tup)

    return sorted(cand, key=lambda x: x[2], reverse=True)[:topn]


def pick_project_id_from_airtable(
    df_view: pd.DataFrame,
    projects_df: pd.DataFrame,
    view_name: str,
    table_name: Optional[str] = None
) -> Tuple[int, str]:
    """
    Ordre de résolution :
      1) colonne 'nom de projet' dans la vue (Nom du projet, Projet, Project name…)
      2) colonne URL → extrait /projects/<id>
      3) fallback depuis nom de vue ou de table
    """
    candidate_label = None

    # (1) colonne ‘nom projet’
    name_cols = [c for c in df_view.columns if re.search(r"(nom.*projet|projet.*nom|project.?name)", str(c), re.I)]
    if name_cols:
        s = df_view[name_cols[0]].dropna().astype(str).str.strip()
        if not s.empty:
            candidate_label = s.value_counts().idxmax()

    # (2) colonne lien → ID
    if candidate_label is None:
        url_cols = [c for c in df_view.columns if re.search(r"(url|lien|link)", str(c), re.I)]
        if url_cols:
            s = df_view[url_cols[0]].dropna().astype(str)
            m = s.str.extract(r"/projects/(\d+)", expand=False).dropna()
            if not m.empty:
                pid = int(m.iloc[0])
                disp = projects_df.loc[projects_df["id"] == pid, "nom_airtable"]
                if not disp.empty:
                    return pid, disp.iloc[0]

    # (3) fallbacks intelligents
    def extract_from_title(txt: str) -> str:
        if not txt:
            return ""
        m = re.search(r"Projet\s+(.+)$", txt, flags=re.I)
        return m.group(1).strip() if m else txt.strip()

    if candidate_label is None:
        looks_like_view_id = bool(re.match(r"^Vue\s+viw[A-Za-z0-9]+$", str(view_name)))
        if table_name and (looks_like_view_id or "Projet" in str(table_name)):
            candidate_label = extract_from_title(str(table_name))
        else:
            candidate_label = extract_from_title(str(view_name))

    st.write("Libellé de croisement Airtable :", f"{candidate_label}")
    cands = guess_candidates_from_label(candidate_label, projects_df, topn=5)

    # Dédupliquer par project_id en gardant le meilleur score
    best = {}
    for pid, disp, score in cands:
        if (pid not in best) or (score > best[pid][1]):
            best[pid] = (disp, float(score))

    cands_unique = [(pid, disp, score) for pid, (disp, score) in best.items()]

    # ⚖️ Tri : d'abord les projets NON "Canceled", puis score décroissant, puis libellé
    def is_canceled(name: str) -> int:
        # 0 = projet normal, 1 = projet "Canceled"
        return 1 if re.search(r"\bcanceled\b", str(name), flags=re.I) else 0

    cands_unique.sort(
        key=lambda x: (
            is_canceled(x[1]),  # non-canceled (0) avant canceled (1)
            -x[2],              # score décroissant
            x[1],               # puis libellé
        )
    )

    options = [
        f"{pid} — {disp} (score {score:.1f})"
        for pid, disp, score in cands_unique
    ] or ["Saisir manuellement"]

    choice = st.selectbox("Confirme le projet exact BO (les plus probables) :", options, index=0)

    if choice == "Saisir manuellement":
        pid = st.number_input("ID projet LPB :", min_value=1, step=1)
        disp = (
            projects_df.loc[projects_df["id"] == pid, "nom_airtable"].iloc[0]
            if pid in projects_df["id"].values
            else f"Projet {int(pid)}"
        )
        return int(pid), disp

    pid = int(re.match(r"^(\d+)\s—", choice).group(1))
    disp = projects_df.loc[projects_df["id"] == pid, "nom_airtable"].iloc[0]
    return pid, disp

    # (3) fallbacks intelligents
    def extract_from_title(txt: str) -> str:
        if not txt:
            return ""
        m = re.search(r"Projet\s+(.+)$", txt, flags=re.I)
        return m.group(1).strip() if m else txt.strip()

    if candidate_label is None:
        looks_like_view_id = bool(re.match(r"^Vue\s+viw[A-Za-z0-9]+$", str(view_name)))
        if table_name and (looks_like_view_id or "Projet" in str(table_name)):
            candidate_label = extract_from_title(str(table_name))
        else:
            candidate_label = extract_from_title(str(view_name))

    st.write("Libellé de croisement Airtable :", f"{candidate_label}")
    cands = guess_candidates_from_label(candidate_label, projects_df, topn=5)

    # Dédupliquer par project_id en gardant le meilleur score
    best = {}
    for pid, disp, score in cands:
        if (pid not in best) or (score > best[pid][1]):
            best[pid] = (disp, float(score))

    cands_unique = [(pid, disp, score) for pid, (disp, score) in best.items()]
    cands_unique.sort(key=lambda x: (-x[2], x[1]))

    options = [f"{pid} — {disp} (score {score:.1f})" for pid, disp, score in cands_unique] or ["Saisir manuellement"]
    choice = st.selectbox("Confirme le projet exact BO (les plus probables) :", options, index=0)

    if choice == "Saisir manuellement":
        pid = st.number_input("ID projet LPB :", min_value=1, step=1)
        disp = projects_df.loc[projects_df["id"] == pid, "nom_airtable"].iloc[0] if pid in projects_df["id"].values else f"Projet {int(pid)}"
        return int(pid), disp

    pid = int(re.match(r"^(\d+)\s—", choice).group(1))
    disp = projects_df.loc[projects_df["id"] == pid, "nom_airtable"].iloc[0]
    return pid, disp

# =======================
# Détection colonnes (emails / prolongation / pouvoir)
# =======================

def detect_email_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.search(r"mail|e-?mail", str(c), re.I)]
    if cols:
        return cols
    # fallback: première colonne contenant '@'
    for c in df.columns:
        try:
            if df[c].astype(str).str.contains("@").any():
                return [c]
        except Exception:
            pass
    return []


def detect_prolongation_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if PROLONG_PATTERN.search(str(c)):
            return c
    return None


def detect_pouvoir_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if POUVOIR_PATTERN.search(str(c)):
            return c
    return None


def standardize_prolongation(val) -> str:
    """
    Prolongation = réponse à :
      - si colonne 'pouvoir' présente :
          « Êtes-vous d'accord pour accorder la prolongation ? »
      - sinon :
          « Êtes-vous d'accord pour appliquer la prolongation,
             avec application des 5% de pénalité ? »
    """
    if pd.isna(val):
        return "Non renseigné"
    s = unidecode(str(val)).strip().lower()
    if s in {"oui", "o", "yes", "y", "true", "1"}:
        return "Oui"
    if s in {"non", "n", "no", "false", "0"}:
        return "Non"
    if re.search(r"ne\s*se\s*pron", s) or s in {"nspp", "ne se prononce pas"}:
        return "Ne se prononce pas"
    return str(val)


def standardize_pouvoir(val) -> str:
    """
    Pouvoir = réponse à :
      « Êtes-vous d'accord pour NE PAS APPLIQUER les pénalités ? »
    """
    if pd.isna(val):
        return "Non renseigné"
    s = unidecode(str(val)).strip().lower()
    if s in {"oui", "o", "yes", "y", "true", "1"}:
        return "Oui"
    if s in {"non", "n", "no", "false", "0"}:
        return "Non"
    if re.search(r"ne\s*se\s*pron", s) or s in {"nspp", "ne se prononce pas"}:
        return "Ne se prononce pas"
    if s in {"", "nan"}:
        return "Non renseigné"
    return str(val)

# =======================
# Construction jeu d’emails Airtable
# =======================

def build_votes_email_flags(
    df_view: pd.DataFrame,
    email_cols: List[str],
    prolong_col: Optional[str],
    pouvoir_col: Optional[str],
) -> pd.DataFrame:
    """
    Agrégé par e-mail :
      - email_normalized
      - email_raw_example
      - prolongation (mode standardisé)
      - pouvoir (mode standardisé)
      - n_occur
    """
    if not email_cols:
        return pd.DataFrame(columns=["email_normalized", "email_raw_example", "prolongation", "pouvoir"])

    melted = []
    for c in email_cols:
        if c not in df_view.columns:
            continue
        s = df_view[[c]].copy()
        s = s.dropna()
        if s.empty:
            continue

        s["email_raw"] = s[c].astype(str)
        s["email_normalized"] = s["email_raw"].map(normalize_email)
        s["source_col"] = c

        if prolong_col and prolong_col in df_view.columns:
            s["prolongation_raw"] = df_view.loc[s.index, prolong_col]
        else:
            s["prolongation_raw"] = None

        if pouvoir_col and pouvoir_col in df_view.columns:
            s["pouvoir_raw"] = df_view.loc[s.index, pouvoir_col]
        else:
            s["pouvoir_raw"] = None

        melted.append(s[["email_raw", "email_normalized", "source_col", "prolongation_raw", "pouvoir_raw"]])

    if not melted:
        return pd.DataFrame(columns=["email_normalized", "email_raw_example", "prolongation", "pouvoir"])

    tmp = pd.concat(melted, ignore_index=True)
    tmp = tmp[tmp["email_normalized"].str.contains("@", na=False)]

    tmp["prolongation_std"] = tmp["prolongation_raw"].apply(standardize_prolongation)
    tmp["pouvoir_std"] = tmp["pouvoir_raw"].apply(standardize_pouvoir)

    agg = (
        tmp
        .groupby("email_normalized", as_index=False)
        .agg(
            n_occur=("email_normalized", "size"),
            email_raw_example=("email_raw", "first"),
            prolongation=("prolongation_std", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else "Non renseigné"),
            pouvoir=("pouvoir_std",      lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else "Non renseigné"),
        )
        .sort_values("email_normalized")
    )
    return agg

# =======================
# UI
# =======================

st.set_page_config(page_title="LPB — Votes ↔ Souscriptions (match & nettoyage)", layout="wide")
st.title("🧱 LPB — Croisement VOTES Airtable ↔ Souscriptions BO")

with st.sidebar:
    st.header("⚙️ Chargement des données")
    st.caption("Lien de vérification Airtable : https://airtable.com/appjOQoptI7Av1obe/tblpoKvFoobl4yej0/viwbPANJvZO7AVX3A?blocks=hide")
    if st.button("🔄 Purger caches"):
        load_projects_df.clear()
        load_subs_for_project.clear()
        st.success("Caches vidés.")
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

with st.spinner("Chargement référentiel projets BO…"):
    projects_df = load_projects_df()

# 1) Lister les vues Airtable contenant 'vote' ET dont la TABLE a prolongation ou pouvoir
if not AIRTABLE_BASE_ID or not AIR_H:
    st.error("Configurez AIRTABLE_TOKEN / AIRTABLE_BASE_ID pour interroger Airtable.")
    st.stop()

tables = list_tables_with_views(AIRTABLE_BASE_ID)
vote_re = re.compile(r"\bvote[s]?\b", re.I)
views = []

for t in tables:
    # Filtre : on ne garde que les tables qui ont prolongation OU pouvoir
    if not table_has_prolongation_or_pouvoir(t):
        continue
    for v in t.get("views", []) or []:
        if vote_re.search(v.get("name", "")):
            views.append((t["name"], t["id"], v["name"], v["id"]))

views = sorted(views, key=lambda x: (x[0].lower(), x[2].lower()))

st.subheader("1) Sélection de la vue Airtable (tableau de votes)")

if not views:
    st.warning("Aucune vue éligible trouvée (nom contenant 'vote' ET table avec colonne 'prolongation' ou 'pouvoir').")
    st.stop()

labels = [f"{tname} • {vname}" for (tname, tid, vname, vid) in views]

q = st.text_input(
    "Rechercher par mot-clé ou coller l’URL d’une vue Airtable (optionnel)",
    placeholder="ex: Solaire ou https://airtable.com/app.../tbl.../viw...",
).strip()

selected_by_url = None
if q.startswith("http"):
    m = re.search(r"airtable\.com/([A-Za-z0-9]+)/([A-Za-z0-9]+)/([A-Za-z0-9]+)", q)
    if m:
        base_from_url, tbl_from_url, view_from_url = m.groups()
        if base_from_url == AIRTABLE_BASE_ID:
            selected_by_url = (tbl_from_url, view_from_url)
        else:
            st.info("⚠️ L’URL ne correspond pas à la base configurée dans ce script (AIRTABLE_BASE_ID).")

if selected_by_url:
    tid, vid = selected_by_url
    tname = next((t for (t, _tid, _v, _vid) in views if _tid == tid), f"Table {tid}")
    vname = next((v for (_t, _tid, v, _vid) in views if _vid == vid), f"Vue {vid}")
else:
    if q and not q.startswith("http"):
        filt_idx = [i for i, lab in enumerate(labels) if q.lower() in lab.lower()]
        if not filt_idx:
            st.info("Aucun résultat pour ce filtre. Affichage de toutes les vues éligibles.")
            options = labels
            idx_map = list(range(len(labels)))
        else:
            options = [labels[i] for i in filt_idx]
            idx_map = filt_idx
    else:
        options = labels
        idx_map = list(range(len(labels)))

    choice = st.selectbox("Choisis un tableau de votes (vue Airtable) :", options, index=0)
    pick = idx_map[options.index(choice)]
    tname, tid, vname, vid = views[pick]

air_url = f"https://airtable.com/{AIRTABLE_BASE_ID}/{tid}/{vid}"
st.markdown(f"🔗 **Lien Airtable :** [{tname} • {vname}]({air_url})")

with st.spinner("Récupération de la vue Airtable…"):
    df_view = flatten(fetch_view_records(AIRTABLE_BASE_ID, tid, vid))

st.write(f"**Vue :** {tname} • {vname} — {len(df_view):,} lignes")
st.dataframe(df_view, use_container_width=True)

# 2) Résolution du projet via BO (name + department)
st.subheader("2) Résolution du projet (référence BO name + department)")
project_id, project_disp = pick_project_id_from_airtable(df_view, projects_df, vname, tname)
proj_url = LPB_PROJECT_URL.format(project_id=project_id)
st.markdown(f"🔗 **Projet choisi :** {project_disp} → [{proj_url}]({proj_url})")

# 3) Emails & réponses Airtable (prolongation + pouvoirs)
st.subheader("3) Emails & Réponses Airtable (prolongation / pénalités)")

# Vérifier que la VUE contient bien AU MOINS une colonne prolongation ou pouvoir
prolong_col = detect_prolongation_column(df_view)
pouvoir_col = detect_pouvoir_column(df_view)

if prolong_col is None and pouvoir_col is None:
    st.error(
        "❌ Cette vue ne contient ni colonne liée à la prolongation ni colonne liée aux pouvoirs.\n\n"
        "Merci de sélectionner une autre vue ou d’ajouter ces colonnes dans la vue Airtable."
    )
    st.stop()

email_cols = detect_email_columns(df_view)
if not email_cols:
    st.error("Impossible de détecter une colonne e-mail dans la vue (aucun '@').")
    st.stop()

picked_em_cols = st.multiselect("Colonnes e-mail à utiliser :", options=email_cols, default=email_cols[:1])

st.caption(
    f"Colonne 'prolongation' détectée : **{prolong_col or 'Aucune'}** — "
    f"colonne 'pouvoir' détectée : **{pouvoir_col or 'Aucune'}**"
)

# Affichage des questions métier
if pouvoir_col is not None:
    st.markdown(
        "**Questions associées :**  \n"
        "- **Prolongation** : _« Êtes-vous d'accord pour accorder la prolongation ? »_  \n"
        "- **Pouvoir (pénalités)** : _« Êtes-vous d'accord pour **NE PAS APPLIQUER** les pénalités ? »_"
    )
else:
    st.markdown(
        "**Question associée :**  \n"
        "- **Prolongation** : _« Êtes-vous d'accord pour appliquer la prolongation, "
        "avec application des 5% de pénalité ? »_"
    )

votes_clean = build_votes_email_flags(df_view, picked_em_cols, prolong_col, pouvoir_col)

# Doublons
dups = votes_clean[votes_clean["n_occur"] > 1].sort_values("n_occur", ascending=False)
st.write("Adresses en doublon (après normalisation) :")
if dups.empty:
    st.success("Aucun doublon détecté ✅")
else:
    st.dataframe(
        dups.rename(columns={"email_raw_example": "Adresse complète", "n_occur": "Nombre de doublons"}),
        use_container_width=True,
    )
    st.download_button(
        "💾 Exporter les doublons (CSV)",
        data=(
            dups[["email_normalized", "email_raw_example", "n_occur"]]
            .rename(columns={"email_raw_example": "adresse_complete", "n_occur": "nombre_doublons"})
            .to_csv(index=False)
            .encode("utf-8")
        ),
        file_name="adresses_doublons.csv",
        mime="text/csv",
    )

st.write(f"Emails uniques dans la vue (après normalisation/déduplication) : **{len(votes_clean):,}**")

# 4) Souscriptions du projet (BO)
st.subheader("4) Souscriptions BO du projet (filtrées par project_id)")
subs = load_subs_for_project(project_id)
st.write(f"Souscriptions actives/historiques (avec e-mail) : **{len(subs):,}**")

# 5) Croisement e-mail (Airtable ↔ BO)
st.subheader("5) Croisement e-mail (Airtable ↔ BO)")

merged = votes_clean.merge(
    subs[["email_normalized", "users_profile_id", "subscription_id", "subscribed_at"]],
    on="email_normalized",
    how="left",
)

# Garder la souscription la plus récente si plusieurs
merged = (
    merged.sort_values(["email_normalized", "subscribed_at"], ascending=[True, False])
          .drop_duplicates("email_normalized")
)

# A. Votants sans souscription
off_proj = merged[merged["subscription_id"].isna()].copy().sort_values("email_normalized")
st.write("Adresses sans souscription détectée :")
if off_proj.empty:
    st.success("Toutes les adresses ont au moins une souscription détectée ✅")
else:
    st.dataframe(
        off_proj[["email_raw_example", "email_normalized", "prolongation", "pouvoir"]]
        .rename(columns={"email_raw_example": "Adresse complète"}),
        use_container_width=True,
    )
    st.download_button(
        "💾 Exporter les adresses sans souscription (CSV)",
        data=(
            off_proj[["email_raw_example", "email_normalized", "prolongation", "pouvoir"]]
            .rename(columns={"email_raw_example": "adresse_complete"})
            .to_csv(index=False)
            .encode("utf-8")
        ),
        file_name=f"adresses_sans_souscription_project_{project_id}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.subheader("📊 KPI résultats")

# Table finale = emails avec souscription
final_tbl = merged[merged["subscription_id"].notna()].copy()
final_tbl = final_tbl[
    [
        "email_raw_example",
        "email_normalized",
        "users_profile_id",
        "subscription_id",
        "subscribed_at",
        "prolongation",
        "pouvoir",
    ]
].rename(columns={"email_raw_example": "Adresse complète"})

n_votes = len(votes_clean)

if not dups.empty:
    n_dups_total = int(dups["n_occur"].sum())
    n_dups_unique = int(len(dups))
    n_dups = int((dups["n_occur"] - 1).sum())
else:
    n_dups_total = n_dups_unique = n_dups = 0

n_with = final_tbl.shape[0]
n_without = off_proj.shape[0]

coverage_rate = (n_with / n_votes * 100) if n_votes else 0.0

total_subs = (
    subs["email_normalized"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .nunique()
)
participation_rate = (n_with / total_subs * 100) if total_subs else 0.0
part_delta = f"{n_with}/{total_subs}" if total_subs else "0/0 (aucune souscription projet)"

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Adresses uniques (vue)", f"{n_votes:,}")
c2.metric("Doublons", f"{n_dups:,}")
c3.metric("Avec souscription", f"{n_with:,}")
c4.metric("Sans souscription", f"{n_without:,}")
c5.metric("Taux de couverture", f"{coverage_rate:.1f}%")
c6.metric("Participation réelle", f"{participation_rate:.1f}%", delta=part_delta)

st.caption(
    "KPI calculés après déduplication. "
    "‘Couverture’ = adresses uniques de la vue avec souscription. "
    "‘Participation’ = souscripteurs ayant répondu / souscripteurs totaux du projet."
)

# ===========================
# 6) Répartition prolongation & pouvoirs (camemberts + verdict)
# ===========================
st.subheader("❓ Répartition des réponses par question (votes légitimes)")

# S'il n'y a aucune souscription → aucun vote légitime → on n'affiche rien
if n_with == 0:
    st.warning(
        "Aucune souscription BO n'a pu être associée aux adresses de la vue. "
        "Il n'y a donc **aucun vote légitime** pour ce projet : "
        "les répartitions de réponses et le verdict ne sont pas calculés."
    )

    st.subheader("🧾 Verdict final")
    st.info(
        "Impossible de calculer un verdict : aucun vote légitime "
        "(0 investisseur avec souscription associée à cette vue)."
    )

    st.markdown(
        "**Table finale — après nettoyage (aucune ligne car aucune souscription associée au projet)**"
    )
    st.dataframe(final_tbl.sort_values("email_normalized"), use_container_width=True)

    st.download_button(
        "💾 Export CSV (table finale)",
        data=final_tbl.to_csv(index=False).encode("utf-8"),
        file_name=f"votes_x_subs_project_{project_id}_final.csv",
        mime="text/csv",
    )
    st.stop()

# À partir d'ici, on sait qu'il existe des votes légitimes
base_df = final_tbl.copy()
acteur_label = "les investisseurs (votes légitimes, e-mails avec souscription)"

st.caption(
    "Les répartitions et le verdict ci-dessous sont calculés **uniquement** "
    "sur les investisseurs identifiés (votes légitimes, associés à une souscription LPB)."
)

import matplotlib.pyplot as plt

def render_pie(counts, title: str):
    """Retourne une figure matplotlib pour un camembert propre."""
    fig, ax = plt.subplots()
    ax.pie(
        counts["Nombre"],
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title(title)
    ax.axis("equal")
    return fig

# ---- Analyse Prolongation ----
counts_pro = None
df_pro = None
q_pro = ""

if prolong_col is not None and "prolongation" in base_df.columns:
    s_pro = base_df["prolongation"].fillna("Non renseigné").astype(str).str.strip()
    counts_pro = s_pro.value_counts()
    df_pro = counts_pro.reset_index()
    df_pro.columns = ["Réponse", "Nombre"]
    df_pro["%"] = (df_pro["Nombre"] / df_pro["Nombre"].sum() * 100).round(1)

    # Question affichée selon présence de pouvoir
    if pouvoir_col is not None:
        q_pro = "Êtes-vous d'accord pour ACCORDER la prolongation ?"
    else:
        q_pro = (
            "Êtes-vous d'accord pour APPLIQUER la prolongation, "
            "avec application des 5% de pénalités ?"
        )

# ---- Analyse Pouvoir (Pénalités) ----
counts_pvr = None
df_pvr = None
q_pvr = ""

if pouvoir_col is not None and "pouvoir" in base_df.columns:
    s_pvr = base_df["pouvoir"].fillna("Non renseigné").astype(str).str.strip()
    counts_pvr = s_pvr.value_counts()
    df_pvr = counts_pvr.reset_index()
    df_pvr.columns = ["Réponse", "Nombre"]
    df_pvr["%"] = (df_pvr["Nombre"] / df_pvr["Nombre"].sum() * 100).round(1)

    q_pvr = "Êtes-vous d'accord pour NE PAS APPLIQUER les pénalités ?"

# ===========================
# 🎨 Affichage côte à côte (camemberts votes légitimes)
# ===========================
col1, col2 = st.columns(2)

if counts_pro is not None:
    with col1:
        st.markdown("### Question — Prolongation")
        st.caption(q_pro)
        st.dataframe(df_pro, use_container_width=True)
        fig_pro = render_pie(df_pro.set_index("Réponse"), "Prolongation")
        st.pyplot(fig_pro, clear_figure=True)

if counts_pvr is not None:
    with col2:
        st.markdown("### Question — Pénalités")
        st.caption(q_pvr)
        st.dataframe(df_pvr, use_container_width=True)
        fig_pvr = render_pie(df_pvr.set_index("Réponse"), "Pénalités")
        st.pyplot(fig_pvr, clear_figure=True)

# ================================
# 7) Verdict final automatisé (sur votes légitimes uniquement)
# ================================
st.subheader("🧾 Verdict final")

def get_yes_no(counts):
    """Retourne uniquement les Oui/Non sous forme (yes, no, total_exprimes)."""
    yes = int(counts.get("Oui", 0))
    no = int(counts.get("Non", 0))
    total_exprimes = yes + no
    return yes, no, total_exprimes

verdict_parts = []

# ---- Verdict sur la prolongation (si disponible) ----
if counts_pro is not None:
    yes_p, no_p, tot_p = get_yes_no(counts_pro)

    if tot_p == 0:
        verdict_parts.append(
            "Sur la question de la prolongation, aucun vote Oui/Non exploitable n'a été exprimé."
        )
    else:
        if pouvoir_col is not None:
            # Cas avec question pénalités séparée -> prolongation simple
            if yes_p > no_p:
                verdict_parts.append(
                    f"Sur la question de la prolongation, {acteur_label} **ACCEPTENT la prolongation** "
                    f"({yes_p} Oui / {tot_p} votes exprimés)."
                )
            elif no_p > yes_p:
                verdict_parts.append(
                    f"Sur la question de la prolongation, {acteur_label} **REFUSENT la prolongation** "
                    f"({no_p} Non / {tot_p} votes exprimés)."
                )
            else:
                verdict_parts.append(
                    f"Sur la question de la prolongation, il y a **égalité parfaite** "
                    f"({yes_p} Oui / {no_p} Non). Décision manuelle nécessaire."
                )
        else:
            # Cas sans colonne pouvoir : prolongation + pénalités
            if yes_p > no_p:
                verdict_parts.append(
                    f"Sur la question « prolongation avec 5% de pénalités », {acteur_label} "
                    f"**ACCEPTENT la prolongation avec pénalités** "
                    f"({yes_p} Oui / {tot_p} votes exprimés)."
                )
            elif no_p > yes_p:
                verdict_parts.append(
                    f"Sur la question « prolongation avec 5% de pénalités », {acteur_label} "
                    f"**REFUSENT la prolongation avec pénalités** "
                    f"({no_p} Non / {tot_p} votes exprimés)."
                )
            else:
                verdict_parts.append(
                    f"Sur la question « prolongation avec 5% de pénalités », il y a **égalité parfaite** "
                    f"({yes_p} Oui / {no_p} Non). Décision manuelle nécessaire."
                )

# ---- Verdict sur les pénalités (si colonne 'pouvoir') ----
if counts_pvr is not None:
    yes_pen, no_pen, tot_pen = get_yes_no(counts_pvr)

    if tot_pen == 0:
        verdict_parts.append(
            "Sur la question des pénalités, aucun vote Oui/Non exploitable n'a été exprimé."
        )
    else:
        if yes_pen > no_pen:
            verdict_parts.append(
                f"Sur la question des pénalités, {acteur_label} "
                f"**VALIDENT la non-application des pénalités** "
                f"({yes_pen} Oui / {tot_pen} votes exprimés)."
            )
        elif no_pen > yes_pen:
            verdict_parts.append(
                f"Sur la question des pénalités, {acteur_label} "
                f"**REFUSENT la non-application des pénalités** "
                f"({no_pen} Non / {tot_pen} votes exprimés)."
            )
        else:
            verdict_parts.append(
                f"Sur la question des pénalités, il y a **égalité parfaite** "
                f"({yes_pen} Oui / {no_pen} Non). Décision manuelle nécessaire."
            )

if not verdict_parts:
    st.markdown("Impossible de calculer un verdict : aucune donnée exploitable.")
else:
    for v in verdict_parts:
        st.markdown("➡️ " + v)

# ================================
# 8) Table finale + export
# ================================
st.markdown("**Table finale — après nettoyage (doublons supprimés) et retrait des invests sans souscription**")
st.dataframe(final_tbl.sort_values("email_normalized"), use_container_width=True)

st.download_button(
    "💾 Export CSV (table finale)",
    data=final_tbl.to_csv(index=False).encode("utf-8"),
    file_name=f"votes_x_subs_project_{project_id}_final.csv",
    mime="text/csv",
)
