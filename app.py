import re
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import streamlit as st    
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from rapidfuzz import process, fuzz
from unidecode import unidecode
import streamlit as st

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False


# --- Secrets ---
AIRTABLE_TOKEN   = st.secrets["airtable"]["token"]
AIRTABLE_BASE_ID = st.secrets["airtable"]["base_id"]
AIR_H = {"Authorization": f"Bearer {AIRTABLE_TOKEN}"}

PG_HOST     = st.secrets["BO"]["host"]
PG_PORT     = st.secrets["BO"]["port"]
PG_DB       = st.secrets["BO"]["db"]
PG_USER     = st.secrets["BO"]["user"]
PG_PASSWORD = st.secrets["BO"]["password"]
PG_SSLMODE  = st.secrets["BO"].get("sslmode", "require")

LPB_PROJECT_URL = "https://app.lapremierebrique.fr/fr/projects/{project_id}"

@st.cache_data(show_spinner=False)
def get_view_last_update(base_id: str, table_id: str, view_name: str) -> datetime:
    """Renvoie la date du record le plus récent dans une vue Airtable."""
    try:
        records = fetch_view_records(base_id, table_id, view_name, page_size=5)
        if not records:
            return datetime.min
        # Certains records peuvent manquer 'createdTime'
        dates = []
        for r in records:
            ct = r.get("createdTime")
            if ct:
                dates.append(datetime.fromisoformat(ct.replace("Z", "+00:00")))
        return max(dates) if dates else datetime.min
    except Exception as e:
        st.warning(f"⏱️ Erreur lors de la récupération de la date pour '{view_name}': {e}")
        return datetime.min

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
# Airtable API helpers
# =======================

def list_tables_with_views(base_id: str) -> List[dict]:
    if not AIR_H:
        return []
    r = requests.get(f"https://api.airtable.com/v0/meta/bases/{base_id}/tables", headers=AIR_H, timeout=30)
    ensure_ok(r)
    return r.json().get("tables", []) or []


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
# BO engine (cache_resource)
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

# =======================
# Chargements BO (cache_data)
# =======================

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
        return pd.DataFrame(columns=["subscription_id", "users_profile_id", "project_id", "subscribed_at", "email_normalized", "email_raw"])
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
# Matching Projet (Airtable → BO name + department)
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
      3) fallback depuis le nom de vue OU (mieux) le nom de table s'il contient 'Projet ...'
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
        # si le nom contient 'Projet ...', on récupère ce qui suit
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
    

    # cands = [(pid, disp, score), ...]  # déjà calculé

    # ✅ Dédupliquer par project_id en gardant le MEILLEUR score
    best = {}
    for pid, disp, score in cands:
        if (pid not in best) or (score > best[pid][1]):
            best[pid] = (disp, float(score))

    cands_unique = [(pid, disp, score) for pid, (disp, score) in best.items()]
    # Tri lisible : score décroissant puis libellé
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
# Détection colonnes (emails & préparation)
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
    # accepte variantes : préparation / prepa / prolongation / NSPP / ne se prononce pas
    pat = re.compile(r"(pr[ée]pa|pr[ée]par|pr[ée]paration|prepa|prolongation|ne\s*se\s*pron|nspp)", re.I)
    for c in df.columns:
        if pat.search(str(c)):
            return c
    return None


def detect_yes_no_column(df: pd.DataFrame) -> Optional[str]:
    """Détecte une colonne binaire Oui/Non (ex: 'prolongation', 'vote', etc.).
    Heuristique: 2 valeurs distinctes (hors NaN) mappables sur Oui/Non, ou nom contenant 'prolong'."""
    prio = [c for c in df.columns if re.search(r"prolong|vote|consent|accord|ok", str(c), re.I)]
    candidates = prio + [c for c in df.columns if c not in prio]
    def _std(v):
        s = unidecode(str(v)).strip().lower()
        if s in {"oui","o","yes","y","true","1"}: return "Oui"
        if s in {"non","n","no","false","0"}: return "Non"
        return None
    for c in candidates:
        vals = pd.Series(df[c].dropna().astype(str).unique()) if c in df.columns else pd.Series([], dtype=str)
        if vals.empty:
            continue
        mapped = vals.map(_std)
        uniq = set(mapped.dropna().unique())
        if uniq.issubset({"Oui","Non"}) and 1 <= len(uniq) <= 2:
            return c
    return None

def standardize_yes_no(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = unidecode(str(val)).strip().lower()
    if s in {"oui","o","yes","y","true","1"}: return "Oui"
    if s in {"non","n","no","false","0"}: return "Non"
    return None

def standardize_prolongation(val) -> str:
    if pd.isna(val):
        return "Non renseigné"
    s = unidecode(str(val)).strip().lower()
    if s in {"oui", "o", "yes", "y"}:
        return "Oui"
    if s in {"non", "n", "no"}:
        return "Non"
    if re.search(r"ne\s*se\s*pron", s) or s in {"nspp", "ne se prononce pas"}:
        return "Ne se prononce pas"
    # valeurs checkbox/array Airtable
    if s in {"true", "1"}:
        return "Oui"
    if s in {"false", "0"}:
        return "Non"
    return str(val)

# =======================
# Construction jeux d'emails côté Airtable
# =======================

def build_votes_email_and_prepa(df_view: pd.DataFrame, email_cols: List[str], prepa_col: Optional[str]) -> pd.DataFrame:
    """
    Retourne un DataFrame avec :
      - email_normalized (unique)
      - email_raw_example (un exemple d'adresse complète telle que vue)
      - prolongation (mode sur les lignes/colonnes, standardisé)
    """
    if not email_cols:
        return pd.DataFrame(columns=["email_normalized", "email_raw_example", "prolongation"]) 

    melted = []
    for c in email_cols:
        s = df_view[[c]].copy()
        s = s.dropna()
        if s.empty:
            continue
        s["email_raw"] = s[c].astype(str)
        s["email_normalized"] = s["email_raw"].map(normalize_email)
        s["source_col"] = c
        if prepa_col and prepa_col in df_view.columns:
            s["prolongation_raw"] = df_view.loc[s.index, prepa_col]
        else:
            s["prolongation_raw"] = None
        melted.append(s[["email_raw", "email_normalized", "source_col", "prolongation_raw"]])

    if not melted:
        return pd.DataFrame(columns=["email_normalized", "email_raw_example", "prolongation"]) 

    tmp = pd.concat(melted, ignore_index=True)
    # garder emails valides
    tmp = tmp[tmp["email_normalized"].str.contains("@", na=False)]

    # standardiser préparation
    tmp["prolongation_std"] = tmp["prolongation_raw"].apply(standardize_prolongation)

    # agrégation par email
    agg = (
        tmp
        .groupby("email_normalized", as_index=False)
        .agg(
            n_occur=("email_normalized", "size"),
            email_raw_example=("email_raw", "first"),
            prolongation=("prolongation_std", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else "Non renseigné"),
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
    st.caption("Lien de vérification Airtable : https://airtable.com/appjOQoptI7Av1obe/tblQPDnKOhYWcgT7f/")
    if st.button("🔄 Purger caches"):
        load_projects_df.clear()
        load_subs_for_project.clear()
        st.success("Caches vidés.")
        try:
            st.rerun()  # Streamlit ≥ 1.30
        except Exception:
            st.experimental_rerun()

with st.spinner("Chargement référentiel projets BO…"):
    projects_df = load_projects_df()

# 1) Lister les vues Airtable contenant 'vote'
if not AIRTABLE_BASE_ID or not AIR_H:
    st.error("Configurez st.secrets['airtable'] (token/base_id) pour interroger Airtable.")
    st.stop()

tables = list_tables_with_views(AIRTABLE_BASE_ID)
vote_re = re.compile(r"\bvote[s]?\b", re.I)
views = []
for t in tables:
    for v in t.get("views", []) or []:
        if vote_re.search(v.get("name", "")):
            views.append((t["name"], t["id"], v["name"], v["id"]))
views = sorted(views, key=lambda x: (x[0].lower(), x[2].lower()))

st.subheader("1) Sélection de la vue Airtable")

if not views:
    st.warning("Aucune vue contenant 'vote' trouvée.")
    st.stop()

# Libellés pour la liste
labels = [f"{tname} • {vname}" for (tname, tid, vname, vid) in views]

# 🔎 Champ recherche OU URL Airtable
q = st.text_input(
    "Rechercher par mot-clé ou coller l’URL d’une vue Airtable (optionnel)",
    placeholder="ex: Solaire ou https://airtable.com/app.../tbl.../viw..."
).strip()

# Si on colle une URL de vue Airtable → on la parse
selected_by_url = None
if q.startswith("http"):
    m = re.search(r"airtable\.com/([A-Za-z0-9]+)/([A-Za-z0-9]+)/([A-Za-z0-9]+)", q)
    # attendu: appXXXXXXXXXXXX / tblXXXXXXXXXXXX / viwXXXXXXXXXXXX
    if m:
        base_from_url, tbl_from_url, view_from_url = m.groups()
        # Si la base correspond et qu'on a table/view → on force la sélection
        if base_from_url == AIRTABLE_BASE_ID:
            selected_by_url = (tbl_from_url, view_from_url)
        else:
            st.info("⚠️ L’URL ne correspond pas à la base configurée dans ce script (AIRTABLE_BASE_ID).")

if selected_by_url:
    # On peut directement utiliser table_id / view_id parsés même si pas listés
    tid, vid = selected_by_url
    tname = next((t for (t, _tid, _v, _vid) in views if _tid == tid), f"Table {tid}")
    vname = next((v for (_t, _tid, v, _vid) in views if _vid == vid), f"Vue {vid}")
else:
    # Sinon, filtrage plein-texte des labels
    if q and not q.startswith("http"):
        filt_idx = [i for i, lab in enumerate(labels) if q.lower() in lab.lower()]
        if not filt_idx:
            st.info("Aucun résultat pour ce filtre. Affichage de toutes les vues.")
            options = labels
            idx_map = list(range(len(labels)))
        else:
            options = [labels[i] for i in filt_idx]
            idx_map = filt_idx
    else:
        options = labels
        idx_map = list(range(len(labels)))

    choice = st.selectbox("Choisis une vue :", options, index=0)
    pick = idx_map[options.index(choice)]
    tname, tid, vname, vid = views[pick]

# 🔗 Lien Airtable direct vers la vue sélectionnée
air_url = f"https://airtable.com/{AIRTABLE_BASE_ID}/{tid}/{vid}"
st.markdown(f"🔗 **Lien Airtable :** [{tname} • {vname}]({air_url})")

with st.spinner("Récupération de la vue Airtable…"):
    df_view = flatten(fetch_view_records(AIRTABLE_BASE_ID, tid, vid))

st.write(f"**Vue :** {tname} • {vname} — {len(df_view):,} lignes")
st.dataframe(df_view, use_container_width=True)


# 1bis) Colonne Oui/Non auto (ex: 'prolongation') — valeurs distinctes + %
bin_col = detect_yes_no_column(df_view)
st.write("Valeurs distinctes initales Oui/Non (colonne détectée)")
if bin_col:
    s = df_view[bin_col]
    total = len(s)
    vc = (
        s.apply(standardize_yes_no)
         .fillna("Non renseigné")
         .value_counts(dropna=False)
         .rename_axis(bin_col)
         .reset_index(name="count")
    )
    vc["pourcentage"] = (vc["count"] / total * 100).round(2)
    st.caption(f"Colonne détectée : **{bin_col}**")
    st.dataframe(vc, use_container_width=True)
else:
    st.info("Aucune colonne binaire Oui/Non détectée (ex: 'prolongation').")

# 2) Résolution du projet via BO (name + department)
st.subheader("2) Résolution du projet (référence BO name + department)")
project_id, project_disp = pick_project_id_from_airtable(df_view, projects_df, vname, tname)
proj_url = LPB_PROJECT_URL.format(project_id=project_id)
st.markdown(f"🔗 **Projet choisi :** {project_disp} → [{proj_url}]({proj_url})")

# 3) Emails & préparation côté Airtable
st.subheader("3) Emails & Réponse côté Airtable (nettoyage + dédup)")
email_cols = detect_email_columns(df_view)
if not email_cols:
    st.error("Impossible de détecter une colonne e-mail dans la vue (aucun '@').")
    st.stop()

picked_em_cols = st.multiselect("Colonnes e-mail à utiliser :", options=email_cols, default=email_cols[:1])
prepa_col = detect_prolongation_column(df_view)
st.caption(f"Colonne de réponse détectée : **{prepa_col or 'Aucune'}** (valeurs standardisées en Oui / Non / Ne se prononce pas / Non renseigné)")

votes_clean = build_votes_email_and_prepa(df_view, picked_em_cols, prepa_col)

# Ajout d'une colonne Oui/Non par email si une colonne binaire existe
if bin_col:
    melted = []
    for c in picked_em_cols:
        if c in df_view.columns:
            s = df_view[[c]].copy().dropna()
            if not s.empty:
                s["email_normalized"] = s[c].map(normalize_email)
                s["oui_non_raw"] = df_view.loc[s.index, bin_col]
                s["oui_non_std"] = s["oui_non_raw"].apply(standardize_yes_no)
                melted.append(s[["email_normalized","oui_non_std"]])
    if melted:
        tmp_yesno = pd.concat(melted, ignore_index=True)
        by_email_yesno = (
            tmp_yesno.dropna(subset=["email_normalized"]) 
                     .groupby("email_normalized", as_index=False)
                     .agg(oui_non=("oui_non_std", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else None))
        )
        votes_clean = votes_clean.merge(by_email_yesno, on="email_normalized", how="left")
else:
    votes_clean["oui_non"] = None

# Doublons (adresse complète + nombre de doublons)
dups = votes_clean[votes_clean["n_occur"] > 1].sort_values("n_occur", ascending=False)
st.write("Adresses en doublon (après normalisation) :")
st.caption("Adresses complètes et nombre d'occurrences par e-mail (toutes colonnes confondues).")
if dups.empty:
    st.success("Aucun doublon détecté ✅")
else:
    st.dataframe(dups.rename(columns={"email_raw_example": "Adresse complète", "n_occur": "Nombre de doublons"}), use_container_width=True)
    st.download_button(
        "💾 Exporter les doublons (CSV)",
        data=dups[["email_normalized", "email_raw_example", "n_occur"]].rename(columns={"email_raw_example": "adresse_complete", "n_occur": "nombre_doublons"}).to_csv(index=False).encode("utf-8"),
        file_name="adresses_doublons.csv",
        mime="text/csv",
    )

st.write(f"Emails uniques dans la vue (après normalisation/déduplication) : **{len(votes_clean):,}**")

# 4) Souscriptions du projet (BO)
st.subheader("4) Souscriptions BO du projet (filtrées par project_id)")
subs = load_subs_for_project(project_id)
st.write(f"Souscriptions actives/historiques (avec e-mail) : **{len(subs):,}**")

# 5) Croisement par e-mail (avec préparation)
st.subheader("5) Croisement e-mail (Airtable ↔ BO) et filtrage")
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
    #st.metric("Nombre sans souscription", f"{len(off_proj):,}")
    st.dataframe(
        off_proj[["email_raw_example", "email_normalized", "prolongation"]]
               .rename(columns={"email_raw_example": "Adresse complète"}),
        use_container_width=True,
    )
    st.download_button(
        "💾 Exporter les adresses sans souscription (CSV)",
        data=off_proj[["email_raw_example", "email_normalized", "prolongation"]]
                .rename(columns={"email_raw_example": "adresse_complete"})
                .to_csv(index=False).encode("utf-8"),
        file_name=f"adresses_sans_souscription_project_{project_id}.csv",
        mime="text/csv",
    )

st.markdown("---") 
st.subheader("📊 KPI résultats")
# B. Table finale = après nettoyage des doublons ET retrait des sans-souscription
final_tbl = merged[merged["subscription_id"].notna()].copy()
final_tbl = final_tbl[[
    "email_raw_example",  # adresse complète
    "email_normalized",
    "users_profile_id",
    "subscription_id",
    "subscribed_at",
    "prolongation",      # Oui / Non / Ne se prononce pas / Non renseigné
]]
final_tbl = final_tbl.rename(columns={"email_raw_example": "Adresse complète"})

# Ajouter la colonne Oui/Non détectée si disponible
if "oui_non" in votes_clean.columns:
    final_tbl = final_tbl.merge(votes_clean[["email_normalized","oui_non"]], on="email_normalized", how="left")
    # réordonner pour afficher oui_non à la fin
    cols = [c for c in final_tbl.columns if c != "oui_non"] + ["oui_non"]
    final_tbl = final_tbl[cols]

# --- KPIs (calculés sur les séries déjà prêtes ci-dessus) ---
n_votes = len(votes_clean)

if not dups.empty:
    n_dups_total = int(dups["n_occur"].sum())
    n_dups_unique = int(len(dups))
    n_dups = int((dups["n_occur"] - 1).sum())  # occurrences en trop (total - uniques)
else:
    n_dups_total = n_dups_unique = n_dups = 0

n_with = final_tbl.shape[0]
n_without = off_proj.shape[0]

# KPI additionnels (final only)
coverage_rate = (n_with / n_votes * 100) if n_votes else 0.0
# --- KPIs (calculés sur les séries déjà prêtes ci-dessus) ---
n_votes = len(votes_clean)

if not dups.empty:
    n_dups_total = int(dups["n_occur"].sum())
    n_dups_unique = int(len(dups))
    n_dups = int((dups["n_occur"] - 1).sum())  # occurrences en trop (total - uniques)
else:
    n_dups_total = n_dups_unique = n_dups = 0

# Souscripteurs apparaissant dans la vue (votants)
n_with = final_tbl.shape[0]
# Adresses uniques de la vue
n_without = off_proj.shape[0]

# --- Couverture (adresses uniques → avec souscription) ---
coverage_rate = (n_with / n_votes * 100) if n_votes else 0.0

# --- Participation (votants / souscripteurs du projet) ---
# Dénominateur = souscripteurs projet (e-mails uniques dans subs)
total_subs = (
    subs["email_normalized"]
    .dropna()
    .astype(str).str.strip().str.lower()
    .nunique()
)
participation_rate = (n_with / total_subs * 100) if total_subs else 0.0

# Si la colonne Oui/Non détectée existe sur la finale, calcule le % Oui (optionnel)
oui_non_rate = None
if "oui_non" in final_tbl.columns:
    s_yn = final_tbl["oui_non"].dropna().astype(str).str.strip()
    if not s_yn.empty:
        oui_non_rate = (s_yn.str.lower().eq("oui").mean() * 100)

# --- Affichage métriques ---
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Adresses uniques", f"{n_votes:,}")
c2.metric("Doublons", f"{n_dups:,}")
c3.metric("Avec souscription", f"{n_with:,}")
c4.metric("Sans souscription", f"{n_without:,}")
c5.metric("Taux de couverture", f"{coverage_rate:.1f}%")
# ⬇️ Remplace l’ancienne métrique “Prépa Oui (finale)”
c6.metric("Participation réelle", f"{participation_rate:.1f}%", delta=f"{n_with}/{total_subs}")

st.caption("KPI calculés après déduplication. Les pourcentages ‘Couverture’ et ‘Participation’ se basent sur adresses e-mail uniques. ‘Participation’ = souscripteurs ayant répondu / souscripteurs totaux du projet.")

# Affiche le KPI Oui/Non final s'il existe (sur une 2e rangée compacte)
# remplacement de l'affichage Oui/Non pour afficher la répartition complète

if "oui_non" in final_tbl.columns:
    s_yn = (
        final_tbl["oui_non"]
        .fillna("Non renseigné")
        .astype(str)
        .str.strip()
        .str.capitalize()
    )
    counts = s_yn.value_counts().reset_index()
    counts.columns = ["Réponse", "Nombre"]
    counts["%"] = (counts["Nombre"] / counts["Nombre"].sum() * 100).round(1)

    cc1, cc2 = st.columns([1, 2])
    cc1.metric("Oui/Non (final) — % Oui", f"{counts.loc[counts['Réponse'] == 'Oui', '%'].iloc[0] if 'Oui' in counts['Réponse'].values else 0.0}%")

    with cc2:
        st.dataframe(counts, use_container_width=True)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.pie(counts["Nombre"], labels=counts["Réponse"], autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.info(f"Graphique non disponible : {e}")

st.markdown("**Table finale — après nettoyage (doublons supprimés) et retrait des invests sans souscription**")
st.dataframe(final_tbl.sort_values("email_normalized"), use_container_width=True)

st.download_button(
    "💾 Export CSV (table finale)",
    data=final_tbl.to_csv(index=False).encode("utf-8"),
    file_name=f"votes_x_subs_project_{project_id}_final.csv",
    mime="text/csv",
)















