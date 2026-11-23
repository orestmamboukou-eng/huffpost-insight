"""Tableau de bord – Projet 3, Groupe 3 (ISSEA ISE3 DSM) : thèmes et sentiments de l'actualité HuffPost.
Lancement : dernière cellule de Projet_G3_fusion.ipynb, ou « streamlit run app_fusion.py »."""
import re
from pathlib import Path
from typing import Any

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

try:
    import plotly.graph_objects as go   # pas plotly.express : il importe xarray, incompatible avec NumPy 2 chez certains
except Exception:
    st.error("Plotly manque : exécutez « pip install plotly » dans l'environnement du notebook, puis relancez.")
    st.stop()

st.set_page_config(page_title="HuffPost Insight", page_icon="📰", layout="wide",
                   initial_sidebar_state="expanded")

DOSSIER = Path(__file__).parent / 'dashboard'
SENTS = ['négatif', 'neutre', 'positif']
C_SENT = {'négatif': '#EF4444', 'neutre': '#CBD5E1', 'positif': '#10B981'}
BLEU, NAVY, ENCRE, GRIS = '#2563EB', '#0B1B47', '#0F172A', '#64748B'
LARGE: dict[str, Any] = ({'width': 'stretch'} if tuple(int(x) for x in st.__version__.split('.')[:2]) >= (1, 50)
                         else {'use_container_width': True})

# =============================================================================== style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, div, span, label, input, button { font-family: 'Inter', sans-serif; }
.stApp { background: #0E1117; }
/* fond sombre : textes hors cartes en clair, textes dans les cartes en foncé */
[data-testid="stWidgetLabel"] p { color: #E2E8F0; }
[class*="st-key-carte"] [data-testid="stWidgetLabel"] p { color: #0F172A; }
[class*="st-key-carte"], .kpi { border-color: #1F2937 !important; box-shadow: 0 6px 22px rgba(0,0,0,.35) !important; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { color: #0F172A; }

/* ---- barre latérale ---- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1B47 0%, #0E2A6B 60%, #0B1B47 100%); }
section[data-testid="stSidebar"] * { color: #E2E8F0; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label { padding: 10px 14px; border-radius: 12px; width: 100%;
    transition: background .15s; margin: 0; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(255,255,255,.08); }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background: #2563EB;
    box-shadow: 0 6px 16px rgba(37,99,235,.45); }
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display: none; }
section[data-testid="stSidebar"] label[data-selected="true"] { background: #2563EB; box-shadow: 0 6px 16px rgba(37,99,235,.45); }
section[data-testid="stSidebar"] div[role="radiogroup"] { width: 100%; }
[data-testid="stSidebarContent"] { padding: 0 10px; }
[data-testid="stDecoration"], [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"] { display: none; }
/* bouton pour rouvrir le menu : visible sur le fond sombre */
[data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { background: #2563EB !important;
    border-radius: 10px; color: #fff !important; box-shadow: 0 4px 14px rgba(37,99,235,.5); }
[data-testid="stSidebarCollapsedControl"] svg, [data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapsedControl"] span, [data-testid="stExpandSidebarButton"] span { color: #fff !important; fill: #fff !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 15.5px; font-weight: 500; }
section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] { background: #fff; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 13px; letter-spacing: 1.5px;
    text-transform: uppercase; color: #94A3B8; }
.logo { display: flex; flex-direction: column; align-items: center; padding: 6px 0 18px; }
.logo .nom { font-size: 27px; font-weight: 800; color: #fff; letter-spacing: -.5px; }
.logo .nom span { color: #60A5FA; }
.logo .dev { font-size: 12.5px; font-style: italic; color: #CBD5E1; margin-top: 2px; }
.side-foot { margin-top: 26px; padding: 16px; border-radius: 14px; background: rgba(255,255,255,.06);
    font-size: 12.5px; line-height: 1.55; color: #CBD5E1; }
.side-foot b { color: #fff; }

/* ---- bandeau ---- */
.hero { position: relative; overflow: hidden; border-radius: 20px; padding: 30px 36px; margin-bottom: 18px;
    background: radial-gradient(circle at 80% 20%, #1D4ED8 0%, rgba(29,78,216,0) 45%),
                linear-gradient(120deg, #0B1B47 0%, #102A78 55%, #1E40AF 100%);
    box-shadow: 0 10px 30px rgba(11,27,71,.25); }
.hero h1, .hero p, .hero .pill { position: relative; z-index: 1; }
.hero p { max-width: 58%; }
.hero h1 { color: #fff; font-size: 34px; font-weight: 800; margin: 0; letter-spacing: -.6px; }
.hero p { color: #C7D2FE; font-size: 16.5px; margin: 6px 0 16px; }
.hero .pill { display: inline-flex; gap: 8px; align-items: center; padding: 8px 16px; border-radius: 999px;
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18); color: #fff; font-size: 14px; font-weight: 600;
    margin-right: 8px; }
.hero svg.deco { position: absolute; right: 18px; bottom: 0; height: 92%; max-width: 42%; opacity: .9; }

/* ---- cartes KPI ---- */
.kpi { background: #fff; border-radius: 18px; padding: 18px 20px 6px; min-height: 158px; border: 1px solid #E6ECF5;
    box-shadow: 0 4px 18px rgba(15,31,75,.06); height: 100%; }
.kpi .top { display: flex; align-items: center; gap: 14px; }
.kpi .ico { width: 52px; height: 52px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex: none; }
.kpi .val { font-size: 27px; font-weight: 800; color: #0F172A; line-height: 1.1; letter-spacing: -.5px; }
.kpi .lab { font-size: 14px; color: #475569; margin-top: 2px; }
.kpi .delta { margin-left: auto; align-self: flex-start; font-size: 13.5px; font-weight: 700; white-space: nowrap; }
.kpi svg.spark { display: block; width: 100%; height: 54px; margin-top: 6px; }

/* ---- cartes graphiques ---- */
[class*="st-key-carte"] { background: #fff; border-radius: 18px !important; border: 1px solid #E6ECF5 !important;
    box-shadow: 0 4px 18px rgba(15,31,75,.06); padding: 16px 18px 8px !important; }
.ctitre { display: flex; align-items: center; gap: 10px; font-size: 18px; font-weight: 700; color: #0F172A; margin-bottom: 2px; }
.ctitre .b { width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center; justify-content: center; }
.csous { font-size: 13.5px; color: #64748B; margin: 0 0 4px 40px; }
.lire { background: #EEF4FF; border-left: 4px solid #2563EB; border-radius: 12px; padding: 12px 16px;
    font-size: 14.5px; color: #1E293B; margin-top: 4px; }

/* ---- liste d'articles ---- */
.art { display: flex; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid #EEF2F7; }
.art:last-child { border-bottom: none; }
.art .pt { width: 34px; height: 34px; border-radius: 50%; flex: none; display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 700; font-size: 13px; }
.art .t { font-size: 14px; font-weight: 600; color: #0F172A; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden; }
.art .d { font-size: 12.5px; color: #64748B; margin-top: 2px; }
.badge { margin-left: auto; flex: none; padding: 5px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 700; }
.b-positif { background: #DCFCE7; color: #047857; } .b-négatif { background: #FEE2E2; color: #B91C1C; }
.b-neutre { background: #F1F5F9; color: #475569; }

/* ---- onglets et widgets de la page ---- */
div[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea { background: #fff !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

ICONES = {  # pictogrammes en trait (24×24)
    'base': '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    'sourire': '<circle cx="12" cy="12" r="9"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><path d="M9 9h.01M15 9h.01"/>',
    'triste': '<circle cx="12" cy="12" r="9"/><path d="M16 16s-1.5-2-4-2-4 2-4 2"/><path d="M9 9h.01M15 9h.01"/>',
    'etoile': '<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/>',
    'courbe': '<path d="M3 17l6-6 4 4 8-8"/><path d="M14 7h7v7"/>',
    'camembert': '<path d="M21 12A9 9 0 1 1 12 3v9z"/><path d="M15 3.5A9 9 0 0 1 20.5 9H15z"/>',
    'barres': '<path d="M4 20V10M10 20V4M16 20v-8M22 20H2"/>',
    'liste': '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
    'carte': '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    'loupe': '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
    'nuage': '<path d="M7 18h10a4 4 0 0 0 0-8 6 6 0 0 0-11.6 1.5A3.5 3.5 0 0 0 7 18z"/>',
    'cible': '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
}


def icone(nom, couleur='#fff', taille=24):
    return (f'<svg width="{taille}" height="{taille}" viewBox="0 0 24 24" fill="none" stroke="{couleur}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONES[nom]}</svg>')


def titre_carte(texte, nom_icone, couleur=BLEU, sous_titre=None):
    st.markdown(f'<div class="ctitre"><span class="b" style="background:{couleur}1A">{icone(nom_icone, couleur, 18)}</span>{texte}</div>'
                + (f'<p class="csous">{sous_titre}</p>' if sous_titre else ''), unsafe_allow_html=True)


def sparkline(valeurs, couleur):
    v = np.asarray(valeurs, dtype=float)
    if len(v) < 2:
        v = np.repeat(v, 2)
    x = np.linspace(0, 300, len(v))
    y = 46 - (v - v.min()) / (np.ptp(v) or 1) * 38
    ligne = ' '.join(f'{a:.1f},{b:.1f}' for a, b in zip(x, y))
    gid = 'g' + couleur.strip('#')
    return (f'<svg class="spark" viewBox="0 0 300 54" preserveAspectRatio="none"><defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{couleur}" stop-opacity=".35"/><stop offset="1" stop-color="{couleur}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="0,54 {ligne} 300,54" fill="url(#{gid})"/><polyline points="{ligne}" fill="none" stroke="{couleur}" stroke-width="2.5"/></svg>')


def carte_kpi(valeur, libelle, nom_icone, couleur, serie, delta=None, delta_bon=True):
    d = ''
    if delta:
        coul = '#059669' if delta_bon else '#DC2626'
        d = f'<div class="delta" style="color:{coul}">{delta}</div>'
    st.markdown(f'<div class="kpi"><div class="top"><div class="ico" style="background:{couleur}">{icone(nom_icone)}</div>'
                f'<div><div class="val">{valeur}</div><div class="lab">{libelle}</div></div>{d}</div>{sparkline(serie, couleur)}</div>',
                unsafe_allow_html=True)


def rgba(hexa, alpha):
    # couleur transparente au format rgba(), accepté par toutes les versions de Plotly
    h = hexa.lstrip('#')
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"


def fmt(n):
    return f"{n:,.0f}".replace(',', ' ')


def pct(x, d=0):
    return f"{x:.{d}f} %".replace('.', ',')


# =============================================================================== graphiques
def mise_en_forme(fig, hauteur=360, legende=True, periodes=False):
    fig.update_layout(
        separators=', ', height=hauteur, margin=dict(l=8, r=8, t=10, b=8), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', size=13, color='#334155'),
        hoverlabel=dict(bgcolor='#0F172A', font=dict(color='#fff', family='Inter'), bordercolor='#0F172A'),
        showlegend=legende, legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, title=None, font=dict(size=12.5)))
    fig.update_xaxes(showgrid=False, linecolor='#E2E8F0', ticks='', title=None)
    fig.update_yaxes(gridcolor='#EEF2F7', zeroline=False, title=None)
    if periodes:
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=CHOIX)
    return fig


def montrer(fig):
    st.plotly_chart(fig, config={'displayModeBar': False}, **LARGE)


# =============================================================================== données
@st.cache_resource
def ressources():
    for p in ['wordnet', 'omw-1.4', 'vader_lexicon']:
        nltk.download(p, quiet=True)
    return joblib.load(DOSSIER / 'modele_fusion.joblib'), SentimentIntensityAnalyzer(), WordNetLemmatizer()


@st.cache_data
def donnees():
    art: pd.DataFrame = pd.read_csv(DOSSIER / 'articles_fusion.csv', parse_dates=['date'], keep_default_na=False,
                                    dtype={'periode': str})
    corpus: pd.DataFrame = pd.read_csv(DOSSIER / 'sentiment_corpus.csv', dtype={'periode': str})
    fiche: pd.DataFrame = pd.read_csv(DOSSIER / 'fiche_themes.csv')
    glob = pd.read_csv(DOSSIER / 'metriques_globales.csv', index_col=0)['Valeur']
    return art, corpus, fiche, glob


@st.cache_data
def nuage(theme, sent, periodes, couleur):
    art = donnees()[0]
    sous = art.loc[(art['theme'] == theme) & (art['sentiment'] == sent) & art['periode'].isin(periodes)]
    freq = pd.Series(' '.join(sous['texte_lem']).split()).value_counts().head(90)
    if len(freq) < 5:
        return None, len(sous)
    from matplotlib.colors import LinearSegmentedColormap
    palette = LinearSegmentedColormap.from_list('p', [couleur + '88', couleur])
    wc = WordCloud(width=900, height=420, background_color='white', colormap=palette, random_state=42,
                   prefer_horizontal=1.0, margin=4).generate_from_frequencies(freq.to_dict())
    return wc.to_array(), len(sous)


if not DOSSIER.exists():
    st.error("Dossier `dashboard/` introuvable : exécutez d'abord la cellule « Export des résultats » du notebook.")
    st.stop()

modele, vader, lemmatiseur = ressources()
art, corpus, fiche, glob = donnees()
ORDRE = modele['ordre']
COUL = dict(zip(ORDRE, ['#EC4899', '#22C55E', '#3B82F6', '#EF4444', '#F59E0B',
                        '#8B5CF6', '#06B6D4', '#92400E', '#64748B', '#F97316']))
PERIODES = sorted(art['periode'].unique())


def solde(s):
    return (s == 'positif').mean() * 100 - (s == 'négatif').mean() * 100


# =============================================================================== barre latérale
PAGES = ["Vue d'ensemble", "Un thème à la loupe", "Sentiments par thème", "Qualité des thèmes",
         "Explorer les articles", "Analyser un texte"]
with st.sidebar:
    st.markdown(f'<div class="logo">{icone("barres", "#60A5FA", 44)}<div class="nom">HuffPost<span>Insight</span></div>'
                '<div class="dev">Analyser · Comprendre · Décider</div></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, label_visibility='collapsed')
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    debut, fin = st.select_slider("Période", options=PERIODES, value=(PERIODES[0], PERIODES[-1]))
    st.markdown('<div class="side-foot"><b>Projet 3 · Groupe 3</b><br>ISSEA · ISE3 · filière DSM<br><br>'
                'Thèmes : lemmatisation → TF-IDF → NMF (K = 10)<br>Sentiment : VADER (seuils ±0,05)<br>'
                'Échantillon stratifié de 25 %</div>', unsafe_allow_html=True)

CHOIX = PERIODES[PERIODES.index(debut): PERIODES.index(fin) + 1]
A: pd.DataFrame = art.loc[art['periode'].isin(CHOIX)]
Cp: pd.DataFrame = corpus.loc[corpus['periode'].isin(CHOIX)]
LIB_PERIODE = debut if debut == fin else f"{debut} → {fin}"


def bandeau(titre, sous_titre, pastilles):
    barres = ''.join(f'<rect x="{30 + i * 34}" y="{150 - h}" width="22" height="{h}" rx="4" fill="#60A5FA" opacity="{0.35 + i * 0.05:.2f}"/>'
                     for i, h in enumerate([30, 44, 38, 60, 52, 78, 70, 96, 88, 120, 112, 140]))
    deco = (f'<svg class="deco" viewBox="0 0 460 160" preserveAspectRatio="xMaxYMax meet">{barres}'
            '<polyline points="30,120 98,104 166,110 234,78 302,62 370,40 440,14" fill="none" stroke="#7DD3FC" stroke-width="4" stroke-linecap="round"/>'
            '<circle cx="440" cy="14" r="7" fill="#fff"/></svg>')
    p = ''.join(f'<span class="pill">{x}</span>' for x in pastilles)
    st.markdown(f'<div class="hero">{deco}<h1>{titre}</h1><p>{sous_titre}</p>{p}</div>', unsafe_allow_html=True)


# =============================================================================== 1. Vue d'ensemble
if page == PAGES[0]:
    bandeau("Cartographie de l'actualité HuffPost",
            "De quoi parle l'actualité, et sur quel ton ? 10 thèmes détectés, une tonalité mesurée article par article.",
            [f"📅 {LIB_PERIODE}", "🧠 TF-IDF + NMF", "💬 VADER"])

    par_per = Cp.pivot_table(index='periode', columns='sentiment', values='n', aggfunc='sum').reindex(CHOIX).fillna(0)
    tot_per = par_per.sum(axis=1)
    part_per = par_per.div(tot_per, axis=0) * 100
    n_tot = int(tot_per.sum())
    pos = par_per['positif'].sum() / n_tot * 100
    neg = par_per['négatif'].sum() / n_tot * 100
    parts_theme = A['theme'].value_counts(normalize=True) * 100
    dominant = parts_theme.index[0]
    serie_dom = [(A.loc[A['periode'] == p, 'theme'] == dominant).mean() * 100 for p in CHOIX]
    ecart = lambda s: s.iloc[-1] - s.iloc[0]

    k = st.columns(4)
    with k[0]:
        carte_kpi(fmt(n_tot), "Articles analysés", 'base', BLEU, tot_per.values)
    with k[1]:
        d = ecart(part_per['positif'])
        carte_kpi(pct(pos), "Articles positifs", 'sourire', '#10B981', part_per['positif'].values,
                  f"{'↑' if d >= 0 else '↓'} {d:+.0f} pts".replace('+-', '−'), d >= 0)
    with k[2]:
        d = ecart(part_per['négatif'])
        carte_kpi(pct(neg), "Articles négatifs", 'triste', '#EF4444', part_per['négatif'].values,
                  f"{'↑' if d >= 0 else '↓'} {d:+.0f} pts", d <= 0)
    with k[3]:
        carte_kpi(pct(parts_theme.iloc[0]), f"Thème n°1 : {dominant}", 'etoile', '#F59E0B', serie_dom)
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    g, d = st.columns([1.65, 1])
    with g:
        with st.container(border=True, key='carte_evol'):
            titre_carte("Évolution du poids des thèmes", 'courbe', BLEU, "Part des articles de chaque période (%) · cliquez la légende pour masquer un thème")
            poids = (A.groupby(['periode', 'theme']).size() / A.groupby('periode').size() * 100).rename('part').reset_index()
            fig = go.Figure()
            for t in ORDRE:
                s = poids[poids['theme'] == t]
                fort = t in ('Trump et Maison Blanche', 'Violences, police, discriminations', 'Cuisine et vie pratique')
                fig.add_trace(go.Scatter(x=s['periode'], y=s['part'], name=t, mode='lines+markers', line_shape='spline',
                                         line=dict(color=COUL[t], width=4 if fort else 2), marker=dict(size=7 if fort else 5),
                                         opacity=1 if fort else .55,
                                         hovertemplate=f'<b>{t}</b><br>%{{x}} : %{{y:.1f}} %<extra></extra>'))
            mise_en_forme(fig, 390, periodes=True)
            fig.update_layout(legend=dict(orientation='v', y=1, x=1.01, yanchor='top', font=dict(size=12)),
                              margin=dict(l=8, r=8, t=10, b=8), hovermode='closest')
            fig.update_yaxes(ticksuffix=' %')
            montrer(fig)
    with d:
        with st.container(border=True, key='carte_donut'):
            titre_carte("Répartition par thème", 'camembert', '#8B5CF6', f"Échantillon · {LIB_PERIODE}")
            dd = parts_theme.reindex(ORDRE).reset_index()
            dd.columns = ['theme', 'part']
            fig = go.Figure(go.Pie(labels=dd['theme'], values=dd['part'], hole=.62, sort=False, direction='clockwise',
                                   marker=dict(colors=[COUL[t] for t in dd['theme']], line=dict(color='#fff', width=2)),
                                   textinfo='none', hovertemplate='<b>%{label}</b><br>%{value:.1f} %<extra></extra>'))
            fig.add_annotation(text=f"<b style='font-size:30px'>10</b><br><span style='color:#64748B'>thèmes</span>",
                               showarrow=False, font=dict(size=14, color=ENCRE))
            mise_en_forme(fig, 390)
            fig.update_layout(legend=dict(orientation='v', x=1.0, y=.5, yanchor='middle', font=dict(size=11.5)))
            montrer(fig)

    g, m, d = st.columns([1.35, .85, 1.1])
    with g:
        with st.container(border=True, key='carte_ton'):
            titre_carte("Tonalité de l'actualité", 'barres', '#EF4444', "Corpus complet · % des articles par période")
            fig = go.Figure()
            for s in SENTS:
                fig.add_trace(go.Bar(x=CHOIX, y=part_per[s], name=s, marker_color=C_SENT[s],
                                     text=[f'{v:.0f}' for v in part_per[s]], textposition='inside',
                                     textfont=dict(color='#fff' if s != 'neutre' else '#334155', size=12),
                                     hovertemplate=f'{s} · %{{x}} : %{{y:.1f}} %<extra></extra>'))
            mise_en_forme(fig, 330, periodes=True)
            fig.update_layout(barmode='stack', bargap=.32)
            fig.update_yaxes(range=[0, 100], ticksuffix=' %')
            montrer(fig)
    with m:
        with st.container(border=True, key='carte_sent'):
            titre_carte("Sentiments", 'cible', '#10B981', LIB_PERIODE)
            vals = [par_per[s].sum() / n_tot * 100 for s in SENTS]
            fig = go.Figure(go.Bar(x=[s.capitalize() for s in SENTS], y=vals, marker=dict(color=[C_SENT[s] for s in SENTS],
                                   cornerradius=10), text=[pct(v, 1) for v in vals], textposition='outside',
                                   textfont=dict(size=14, color=ENCRE), hovertemplate='%{x} : %{y:.1f} %<extra></extra>'))
            mise_en_forme(fig, 330, legende=False)
            fig.update_yaxes(visible=False, range=[0, max(vals) * 1.25])
            fig.update_layout(bargap=.35)
            montrer(fig)
    with d:
        with st.container(border=True, key='carte_recents'):
            titre_carte("Derniers articles de la période", 'liste', BLEU)
            lignes = ''
            for _, r in A.nlargest(6, 'date').iterrows():
                th, se = str(r['theme']), str(r['sentiment'])
                dt = '/'.join(reversed(str(r['date'])[:10].split('-')))
                c = COUL[th]
                init = ''.join(w[0] for w in re.findall(r'\w+', th)[:2]).upper()
                lignes += (f'<div class="art"><div class="pt" style="background:{c}">{init}</div><div style="min-width:0">'
                           f'<div class="t">{r["headline"]}</div><div class="d">{dt} · {th}</div></div>'
                           f'<span class="badge b-{se}">{se.capitalize()}</span></div>')
            st.markdown(lignes, unsafe_allow_html=True)

    st.markdown('<div class="lire"><b>À retenir.</b> <i>Trump et Maison Blanche</i> passe de 2 % des articles en 2012 à 26 % en 2018, '
                'pendant que <i>Cuisine</i>, <i>Santé</i> et <i>Famille</i> s\'effondrent. La part d\'articles négatifs passe de 25 % à 47 %.</div>',
                unsafe_allow_html=True)

# =============================================================================== 2. Un thème à la loupe
elif page == PAGES[1]:
    theme = st.selectbox("Thème", ORDRE, key='theme_loupe')
    c = COUL[theme]
    T: pd.DataFrame = A.loc[A['theme'] == theme]
    f = fiche.set_index('Thème').loc[theme]
    bandeau(theme, f"Mots clés : {f['Mots qui définissent le thème']}", [f"📅 {LIB_PERIODE}", f"🏷️ {len(T):,} articles".replace(',', ' ')])

    evo = pd.DataFrame({'part': A.groupby('periode')['theme'].apply(lambda t: (t == theme).mean() * 100),
                        'solde': T.groupby('periode')['sentiment'].apply(solde)}).reindex(CHOIX)
    k = st.columns(4)
    with k[0]:
        carte_kpi(pct(len(T) / max(len(A), 1) * 100, 1), "Part des articles", 'camembert', c, evo['part'].fillna(0))
    with k[1]:
        carte_kpi(f"{f['Cohérence C_v']:.2f}".replace('.', ','), "Cohérence C_v", 'cible', '#8B5CF6', [f['Cohérence C_v']] * 2)
    with k[2]:
        carte_kpi(pct(f['Articles nets'] * 100), "Articles nets (dominance ≥ 0,5)", 'etoile', '#F59E0B', [f['Articles nets']] * 2)
    with k[3]:
        s_ = solde(T['sentiment'])
        carte_kpi(f"{s_:+.0f}", "Solde d'opinion", 'sourire' if s_ >= 0 else 'triste',
                  '#10B981' if s_ >= 0 else '#EF4444', evo['solde'].fillna(0))
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    g, d = st.columns(2)
    with g:
        with st.container(border=True, key='carte_mots'):
            titre_carte("Les mots qui le définissent", 'liste', c, "Poids dans la matrice H de la NMF (relatif au 1er mot)")
            idx = [i for i, n in modele['noms'].items() if n == theme][0]
            H = modele['nmf'].components_
            vocab = modele['tfidf'].get_feature_names_out()
            top = H[idx].argsort()[::-1][:12][::-1]
            fig = go.Figure(go.Bar(x=H[idx, top] / H[idx, top].max(), y=vocab[top], orientation='h',
                                   marker=dict(color=c, cornerradius=6), hovertemplate='%{y} : %{x:.2f}<extra></extra>'))
            mise_en_forme(fig, 380, legende=False)
            fig.update_xaxes(visible=False)
            montrer(fig)
    with d:
        with st.container(border=True, key='carte_rub'):
            titre_carte("Rubriques d'origine", 'camembert', BLEU, "Rubriques HuffPost des articles du thème")
            rub = T['category'].value_counts(normalize=True).mul(100)
            rub = pd.concat([rub.head(6), pd.Series({'Autres': rub.iloc[6:].sum()})])
            fig = go.Figure(go.Pie(labels=rub.index, values=rub.values, hole=.58, sort=False,
                                   marker=dict(colors=['#08306B', '#08519C', '#2171B5', '#4292C6', '#6BAED6', '#9ECAE1', '#C6DBEF'], line=dict(color='#fff', width=2)),
                                   textinfo='percent', textfont=dict(size=12), hovertemplate='%{label} : %{value:.1f} %<extra></extra>'))
            mise_en_forme(fig, 380)
            fig.update_layout(legend=dict(orientation='v', x=1, y=.5, yanchor='middle'))
            montrer(fig)

    g, d = st.columns(2)
    for col, sent in [(g, 'positif'), (d, 'négatif')]:
        with col:
            with st.container(border=True, key=f'carte_nuage_{sent}'):
                image, n = nuage(theme, sent, tuple(CHOIX), C_SENT[sent] if sent == 'négatif' else '#059669')
                titre_carte(f"Articles {sent}s · {fmt(n)}", 'nuage', C_SENT[sent])
                if image is None:
                    st.caption("Trop peu d'articles sur cette période.")
                else:
                    st.image(image, **LARGE)

    with st.container(border=True, key='carte_evo_theme'):
        titre_carte("Place et tonalité du thème dans le temps", 'courbe', c)
        g, d = st.columns(2)
        with g:
            fig = go.Figure(go.Scatter(x=evo.index, y=evo['part'], mode='lines+markers', line_shape='spline', fill='tozeroy',
                                       line=dict(color=c, width=3.5), fillcolor=rgba(c, .13), marker=dict(size=8),
                                       hovertemplate='%{x} : %{y:.1f} %<extra></extra>'))
            mise_en_forme(fig, 280, legende=False, periodes=True)
            fig.update_layout(margin=dict(t=34))
            fig.update_yaxes(ticksuffix=' %')
            fig.add_annotation(text="Part des articles", x=0, y=1.08, xref='paper', yref='paper', showarrow=False,
                               font=dict(size=13, color=GRIS), xanchor='left')
            montrer(fig)
        with d:
            cols = ['#10B981' if v >= 0 else '#EF4444' for v in evo['solde'].fillna(0)]
            fig = go.Figure(go.Bar(x=evo.index, y=evo['solde'], marker=dict(color=cols, cornerradius=6),
                                   text=[f'{v:+.0f}' for v in evo['solde'].fillna(0)], textposition='outside',
                                   hovertemplate='%{x} : %{y:+.0f}<extra></extra>'))
            mise_en_forme(fig, 280, legende=False, periodes=True)
            fig.update_layout(margin=dict(t=34))
            fig.add_hline(y=0, line_color='#94A3B8', line_width=1)
            fig.add_annotation(text="Solde d'opinion (% positifs − % négatifs)", x=0, y=1.08, xref='paper', yref='paper',
                               showarrow=False, font=dict(size=13, color=GRIS), xanchor='left')
            montrer(fig)

    with st.container(border=True, key='carte_exemples'):
        titre_carte("Articles les plus récents du thème", 'liste', c)
        st.dataframe(T.nlargest(10, 'date')[['date', 'headline', 'category', 'sentiment', 'compound', 'link']],
                     hide_index=True, **LARGE,
                     column_config={'link': st.column_config.LinkColumn('Lien', display_text='ouvrir'),
                                    'date': st.column_config.DateColumn('Date', format='DD/MM/YYYY'),
                                    'headline': st.column_config.TextColumn('Titre', width='large'),
                                    'category': 'Rubrique', 'sentiment': 'Sentiment',
                                    'compound': st.column_config.NumberColumn('Score VADER', format='%.2f')})

# =============================================================================== 3. Sentiments par thème
elif page == PAGES[2]:
    bandeau("Sentiments par thème", "Chaque thème a sa tonalité, et elle évolue dans le temps.",
            [f"📅 {LIB_PERIODE}", "Solde = % positifs − % négatifs"])
    rep = pd.DataFrame((A.groupby('theme')['sentiment'].value_counts(normalize=True) * 100).unstack()).reindex(columns=SENTS).fillna(0)
    rep['solde'] = rep['positif'] - rep['négatif']
    rep = rep.sort_values('solde')

    g, d = st.columns([1, 1.35])
    with g:
        with st.container(border=True, key='carte_diverge'):
            titre_carte("Tonalité de chaque thème", 'barres', '#EF4444', "Négatifs à gauche, positifs à droite (% des articles)")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=rep.index, x=-rep['négatif'], orientation='h', name='négatif', marker=dict(color=C_SENT['négatif'], cornerradius=5),
                                 customdata=rep['négatif'], hovertemplate='%{y}<br>négatifs : %{customdata:.1f} %<extra></extra>'))
            fig.add_trace(go.Bar(y=rep.index, x=rep['positif'], orientation='h', name='positif', marker=dict(color=C_SENT['positif'], cornerradius=5),
                                 hovertemplate='%{y}<br>positifs : %{x:.1f} %<extra></extra>'))
            mise_en_forme(fig, 470)
            fig.update_layout(barmode='relative', bargap=.3)
            fig.update_xaxes(tickvals=[-60, -40, -20, 0, 20, 40, 60], ticktext=['60 %', '40 %', '20 %', '0', '20 %', '40 %', '60 %'],
                             zeroline=True, zerolinecolor='#94A3B8')
            montrer(fig)
    with d:
        with st.container(border=True, key='carte_heat'):
            titre_carte("Solde d'opinion par thème et par période", 'carte', '#F59E0B', "Vert : plus de positifs · rouge : plus de négatifs")
            sol = pd.DataFrame(pd.DataFrame(A.groupby(['theme', 'periode'])['sentiment'].apply(solde).unstack()).reindex(ORDRE)[CHOIX])
            lim = float(sol.abs().max().max())
            fig = go.Figure(go.Heatmap(z=sol.values, x=CHOIX, y=sol.index, zmin=-lim, zmax=lim, xgap=3, ygap=3,
                                       colorscale=[[0, '#DC2626'], [.5, '#FEF9C3'], [1, '#059669']],
                                       text=[[f'{v:+.0f}' for v in r] for r in sol.values], texttemplate='%{text}',
                                       textfont=dict(size=12.5), colorbar=dict(thickness=10, outlinewidth=0),
                                       hovertemplate='%{y}<br>%{x} : %{z:+.0f}<extra></extra>'))
            mise_en_forme(fig, 470, legende=False, periodes=True)
            fig.update_yaxes(autorange='reversed', gridcolor='rgba(0,0,0,0)')
            fig.update_xaxes(side='top')
            montrer(fig)

    with st.container(border=True, key='carte_compare'):
        titre_carte("Comparer l'évolution de plusieurs thèmes", 'courbe', BLEU)
        choix = st.multiselect("Thèmes", ORDRE, default=['Trump et Maison Blanche', 'Violences, police, discriminations',
                                                         'Cuisine et vie pratique', 'Famille et enfants'], label_visibility='collapsed')
        if choix:
            fig = go.Figure()
            for t in choix:
                fig.add_trace(go.Scatter(x=CHOIX, y=sol.loc[t], name=t, mode='lines+markers', line_shape='spline',
                                         line=dict(color=COUL[t], width=3.5), marker=dict(size=8),
                                         hovertemplate=f'<b>{t}</b><br>%{{x}} : %{{y:+.0f}}<extra></extra>'))
            fig.add_hrect(y0=0, y1=max(60, float(np.nanmax(sol.values))), fillcolor='#10B981', opacity=.05, line_width=0)
            fig.add_hrect(y0=min(-60, float(np.nanmin(sol.values))), y1=0, fillcolor='#EF4444', opacity=.05, line_width=0)
            fig.add_hline(y=0, line_color='#94A3B8', line_dash='dot')
            mise_en_forme(fig, 360, periodes=True)
            montrer(fig)
    st.markdown('<div class="lire"><b>À retenir.</b> <i>Cuisine</i>, <i>Célébrités</i> et <i>Famille</i> restent positifs chaque année ; '
                '<i>Violences, police, discriminations</i> est négatif chaque année et s\'aggrave. De 2012 à 2018, le solde baisse dans les dix thèmes.</div>',
                unsafe_allow_html=True)

# =============================================================================== 4. Qualité
elif page == PAGES[3]:
    bandeau("Qualité des thèmes", "Les 10 thèmes sont-ils cohérents et bien séparés ?", ["Cohérence C_v", "Silhouette", "Exclusivité"])
    k = st.columns(4)
    with k[0]:
        carte_kpi(f"{glob.iloc[3]:.3f}".replace('.', ','), "Silhouette (hasard : −0,018)", 'cible', BLEU, [glob.iloc[4], glob.iloc[3]],
                  f"↑ {glob.iloc[3] - glob.iloc[4]:+.2f}".replace('.', ','))
    with k[1]:
        carte_kpi(f"{glob.iloc[0]:.2f}".replace('.', ','), "Cohérence C_v moyenne", 'etoile', '#8B5CF6', fiche['Cohérence C_v'].values)
    with k[2]:
        carte_kpi(pct(glob.iloc[1] * 100), "Mots uniques entre thèmes", 'liste', '#10B981', fiche['Exclusivité'].values)
    with k[3]:
        carte_kpi(f"{glob.iloc[2]:.3f}".replace('.', ','), "Similarité moyenne entre thèmes", 'carte', '#F59E0B', fiche['Similarité max'].values)
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    g, d = st.columns([1.15, 1])
    with g:
        with st.container(border=True, key='carte_bulles'):
            titre_carte("Cohérence et exclusivité", 'cible', '#8B5CF6', "Taille des bulles : part des articles")
            fig = go.Figure()
            for _, r in fiche.iterrows():
                fig.add_trace(go.Scatter(x=[r['Cohérence C_v']], y=[r['Exclusivité']], mode='markers+text', name=r['Thème'],
                                         marker=dict(size=r['Part des articles'] * 420, color=COUL[r['Thème']], opacity=.85,
                                                     line=dict(color='#fff', width=2)),
                                         text=[str(r['Thème']).split(',')[0]], textposition='top center', textfont=dict(size=11.5),
                                         hovertemplate=f"<b>{r['Thème']}</b><br>C_v %{{x:.2f}} · exclusivité %{{y:.2f}}<extra></extra>"))
            mise_en_forme(fig, 440, legende=False)
            fig.update_xaxes(title='Cohérence C_v', showgrid=True, gridcolor='#EEF2F7')
            fig.update_yaxes(title='Exclusivité')
            montrer(fig)
    with d:
        with st.container(border=True, key='carte_cv'):
            titre_carte("Cohérence de chaque thème", 'barres', '#10B981', "C_v : plus c'est haut, plus les mots du thème vont ensemble")
            fc = fiche.sort_values('Cohérence C_v')
            fig = go.Figure(go.Bar(x=fc['Cohérence C_v'], y=fc['Thème'], orientation='h',
                                   marker=dict(color=[COUL[t] for t in fc['Thème']], cornerradius=6),
                                   text=[f'{v:.2f}'.replace('.', ',') for v in fc['Cohérence C_v']], textposition='outside',
                                   hovertemplate='%{y} : %{x:.2f}<extra></extra>'))
            mise_en_forme(fig, 440, legende=False)
            fig.add_vline(x=float(glob.iloc[0]), line_dash='dot', line_color='#64748B',
                          annotation_text='moyenne', annotation_position='top')
            fig.update_xaxes(visible=False, range=[0, 0.78])
            montrer(fig)
    with st.container(border=True, key='carte_fiche'):
        titre_carte("Fiche de chaque thème", 'liste', BLEU)
        st.dataframe(
            fiche.set_index('Thème')[['Part des articles', 'Cohérence C_v', 'Exclusivité', 'Articles nets',
                                      'Similarité max', 'Mots qui définissent le thème']],
            height=410, **LARGE,
            column_config={
                'Part des articles': st.column_config.ProgressColumn('Part', format='%.3f', min_value=0, max_value=0.2),
                'Cohérence C_v': st.column_config.ProgressColumn('C_v', format='%.2f', min_value=0, max_value=1),
                'Exclusivité': st.column_config.ProgressColumn(format='%.2f', min_value=0, max_value=1),
                'Articles nets': st.column_config.ProgressColumn('Nets', format='%.2f', min_value=0, max_value=1),
                'Similarité max': st.column_config.NumberColumn('Sim. max', format='%.2f'),
                'Mots qui définissent le thème': st.column_config.TextColumn('Mots', width='large')})


# =============================================================================== 5. Explorer
elif page == PAGES[4]:
    bandeau("Explorer les articles", "Filtrez l'échantillon par thème, sentiment, rubrique ou mot-clé.", [f"📅 {LIB_PERIODE}"])
    with st.container(border=True, key='carte_filtres'):
        c = st.columns([1.3, 1, 1, 1.4])
        f_theme = c[0].multiselect("Thème", ORDRE)
        f_sent = c[1].multiselect("Sentiment", SENTS)
        f_rub = c[2].multiselect("Rubrique", sorted(A['category'].unique()))
        f_mot = c[3].text_input("Mot dans le titre ou le résumé", placeholder="ex. climate")
    E: pd.DataFrame = A
    if f_theme:
        E = E.loc[E['theme'].isin(f_theme)]
    if f_sent:
        E = E.loc[E['sentiment'].isin(f_sent)]
    if f_rub:
        E = E.loc[E['category'].isin(f_rub)]
    if f_mot:
        motif = re.escape(f_mot.strip())
        E = E.loc[E['headline'].str.contains(motif, case=False) | E['short_description'].str.contains(motif, case=False)]
    with st.container(border=True, key='carte_table'):
        titre_carte(f"{fmt(len(E))} articles", 'liste', BLEU, "500 plus récents affichés")
        st.dataframe(
            E.sort_values('date', ascending=False)[['date', 'headline', 'theme', 'category', 'sentiment', 'compound', 'link']].head(500),
            hide_index=True, height=560, **LARGE,
            column_config={'date': st.column_config.DateColumn('Date', format='DD/MM/YYYY'),
                           'headline': st.column_config.TextColumn('Titre', width='large'),
                           'theme': 'Thème', 'category': 'Rubrique', 'sentiment': 'Sentiment',
                           'compound': st.column_config.ProgressColumn('Score VADER', format='%.2f', min_value=-1, max_value=1),
                           'link': st.column_config.LinkColumn('Lien', display_text='ouvrir')})

# =============================================================================== 6. Analyser un texte
else:
    bandeau("Analyser un texte", "Collez un titre ou un court article en anglais : thème dominant et tonalité.", ["NMF", "VADER"])
    g, d = st.columns([1, 1.3])
    with g:
        with st.container(border=True, key='carte_saisie'):
            titre_carte("Texte à analyser", 'loupe', BLEU)
            texte = st.text_area("Texte", height=170, label_visibility='collapsed',
                                 value="Police officers accused of excessive force after a shooting in Chicago")
            lancer = st.button("Analyser", type="primary", **LARGE)
    if texte.strip():
        t = texte.lower()
        t = re.sub(r"http\S+", " ", t)
        t = re.sub(r"['’]s\b", " ", t)
        t = re.sub(r"[^a-z ]", " ", t)
        mots = [lemmatiseur.lemmatize(m) for m in t.split() if m not in modele['stop'] and len(m) > 2]
        w = modele['nmf'].transform(modele['tfidf'].transform([' '.join(mots)]))[0]
        score = vader.polarity_scores(texte)['compound']
        sent = 'positif' if score >= 0.05 else 'négatif' if score <= -0.05 else 'neutre'
        with g:
            with st.container(border=True, key='carte_jauge'):
                titre_carte("Tonalité (VADER)", 'cible', C_SENT[sent], f"Sentiment : {sent}")
                fig = go.Figure(go.Indicator(
                    mode='gauge+number', value=score, number=dict(valueformat='+.2f', font=dict(size=40, color=ENCRE)),
                    gauge=dict(axis=dict(range=[-1, 1], tickvals=[-1, -.5, 0, .5, 1], ticktext=['−1', '−0,5', '0', '+0,5', '+1']),
                               bar=dict(color=ENCRE, thickness=.25),
                               steps=[dict(range=[-1, -.05], color='#FECACA'), dict(range=[-.05, .05], color='#E2E8F0'),
                                      dict(range=[.05, 1], color='#BBF7D0')], borderwidth=0)))
                mise_en_forme(fig, 230, legende=False)
                montrer(fig)
        with d:
            with st.container(border=True, key='carte_resultat'):
                if w.sum() == 0:
                    titre_carte("Thème", 'camembert', '#8B5CF6')
                    st.warning("Aucun mot du texte n'est connu du modèle : thème impossible à déterminer.")
                else:
                    p = pd.Series(w / w.sum() * 100, index=[modele['noms'][i] for i in range(len(w))]).sort_values()
                    titre_carte(f"Thème dominant : {p.index[-1]}", 'camembert', COUL[p.index[-1]], "Poids de chaque thème dans le texte")
                    fig = go.Figure(go.Bar(x=p.values, y=p.index, orientation='h', text=[pct(v, 0) for v in p.values], textposition='outside',
                                           marker=dict(color=[COUL[x] for x in p.index], cornerradius=6),
                                           hovertemplate='%{y} : %{x:.1f} %<extra></extra>'))
                    mise_en_forme(fig, 430, legende=False)
                    fig.update_xaxes(visible=False, range=[0, p.max() * 1.18])
                    montrer(fig)
                st.caption("Mots retenus après nettoyage et lemmatisation : " + (', '.join(mots) or '—'))
