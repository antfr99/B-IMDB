import io
import urllib.request
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="My IMDb Ratings", page_icon="🎬", layout="wide")

GOLD = "#F5C518"   # IMDb yellow
TEAL = "#4FA88F"

# ── Catalog location (public GitHub repo) ─────────────────────────────────────
GITHUB_USER, GITHUB_REPO, GITHUB_BRANCH = "antfr99", "B-IMDB", "main"
CATALOG_URL = (f"https://raw.githubusercontent.com/"
               f"{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/imdb_movies.csv")

st.markdown("""
<style>
.stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {background:#000;}
[data-testid="stMetricValue"] {color:#F5C518;}
.info-box {background:#111; border:1px solid #333; border-radius:8px;
    padding:16px 20px; margin-bottom:16px; font-size:0.88rem; color:#aaa; line-height:1.6;}
.info-box a {color:#F5C518; text-decoration:none;}
.info-box a:hover {text-decoration:underline;}
.info-box code {color:#F5C518; background:#222; padding:1px 5px; border-radius:3px;}
.badge {display:inline-block; background:#F5C518; color:#000; border-radius:4px;
    padding:2px 8px; font-size:0.78rem; font-weight:700; margin-right:6px;}
.dash {background:linear-gradient(160deg,#3a2f05 0%, #000 70%);
    border:1px solid #F5C518; border-radius:20px; padding:32px; margin:8px 0 24px;}
.dash h2 {text-align:center; letter-spacing:.28em; color:#F5C518; font-size:.85rem;
    font-weight:700; margin:0 0 4px;}
.dash .head {text-align:center; font-size:2.1rem; font-weight:800; color:#fff; margin:8px 0 24px;}
.dash .row3 {display:flex; justify-content:space-around; text-align:center; margin-bottom:8px;}
.dash .big {font-size:2rem; font-weight:800; color:#F5C518; line-height:1;}
.dash .lbl {font-size:.72rem; letter-spacing:.12em; color:#aaa; margin-top:6px;}
.dash hr {border:none; border-top:1px solid #5c4a0a; margin:20px 0;}
.dash .stat-lbl {font-size:.7rem; letter-spacing:.16em; color:#F5C518; font-weight:700; margin-top:14px;}
.dash .stat-val {font-size:1.25rem; font-weight:700; color:#fff;}
</style>""", unsafe_allow_html=True)

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading movie catalog from GitHub…")
def load_catalog() -> pd.DataFrame:
    try:
        with urllib.request.urlopen(CATALOG_URL, timeout=20) as r:
            cat = pd.read_csv(io.BytesIO(r.read()))
    except Exception:
        return pd.DataFrame()
    # The CSV ships with human-readable headers ("Movie ID", "Movie URL", …).
    # Map them to the lowercase names the rest of the app uses.
    cat = cat.rename(columns={c: c.strip() for c in cat.columns})
    cat = cat.rename(columns={
        "Movie ID": "movie_id", "Title": "title", "Year": "year",
        "Runtime": "runtime", "Rating": "rating", "Votes": "votes",
        "Genre": "genre", "Director": "director", "Movie URL": "movie_url",
    })
    return cat

def split_multi(series: pd.Series) -> pd.Series:
    """Explode a comma-separated column into individual trimmed values."""
    return (series.dropna().astype(str)
            .str.split(",").explode().str.strip().replace("", pd.NA).dropna())

@st.cache_data(show_spinner="Reading your ratings…")
def load_ratings(blob: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(blob))
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # Parse dates / numerics defensively
    for col in ["Date Rated", "Release Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["Your Rating", "IMDb Rating", "Runtime (mins)", "Year", "Num Votes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ── Sidebar: how-to + upload ──────────────────────────────────────────────────
st.sidebar.title("🎬 IMDb Ratings")
st.sidebar.markdown("""
<div class="info-box">
<span class="badge">YOUR RATINGS</span><b>Where to get your ratings file</b><br><br>
1. Go to <a href="https://www.imdb.com/" target="_blank">IMDb.com</a> and sign in.<br>
2. Click your name → <b>Your Ratings</b>.<br>
3. Click the <b>⋯ (three dots)</b> menu → <b>Export</b>.<br>
4. IMDb emails / downloads a <code>ratings.csv</code> file.<br><br>
Upload that file below. It stays in memory for this session — nothing is stored.
</div>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Upload your IMDb ratings CSV", type="csv",
    help="The ratings.csv you exported from IMDb → Your Ratings → Export.")

st.sidebar.markdown("""
<div class="info-box">
<span class="badge">CATALOG</span><b>The reference catalog</b><br><br>
The <code>imdb_movies.csv</code> catalog (loaded from GitHub) is used for the
<b>Discover</b> tab. It contains, by design:<br><br>
• Movies only<br>
• Released 1950 onward<br>
• IMDb rating above 6.8<br>
• Adult titles excluded<br>
• Director names included<br>
• IMDb URL included
</div>
""", unsafe_allow_html=True)

catalog = load_catalog()

if uploaded is None:
    st.title("My IMDb ratings")
    st.markdown("""
<div class="info-box">
<span class="badge">GET STARTED</span><br><br>
<b>Step 1 — Export your ratings from IMDb</b><br>
Sign in at <a href="https://www.imdb.com/" target="_blank">IMDb.com</a> →
your name → <b>Your Ratings</b> → <b>⋯</b> → <b>Export</b>. You get a
<code>ratings.csv</code>.<br><br>
<b>Step 2 — Upload it</b><br>
Use the <b>Upload</b> button in the sidebar. Everything runs in your browser
session; your file is never saved.<br><br>
The app analyses your taste, compares it to the IMDb crowd, and suggests
highly-rated films you haven't seen yet from a curated catalog
(<b>movies only, 1950+, IMDb rating above 6.8, no adult titles</b>).
</div>
""", unsafe_allow_html=True)
    if catalog.empty:
        st.warning("Note: the reference catalog couldn't be loaded from GitHub right now — "
                   "the Discover tab will be unavailable, but your own ratings analysis still works.")
    st.stop()

df = load_ratings(uploaded.getvalue())

# ── Sidebar filters (apply to YOUR RATINGS) ───────────────────────────────────
st.sidebar.header("Filter my ratings")

movies_only = st.sidebar.checkbox("Movies only (exclude TV)", value=True,
    help="Your export may include TV episodes and series. Catalog is movies only.")
if movies_only and "Title Type" in df.columns:
    df = df[df["Title Type"].astype(str).str.contains("movie", case=False, na=False)]

# Title search
title_q = st.sidebar.text_input("Title contains")
if title_q and "Title" in df.columns:
    df = df[df["Title"].astype(str).str.contains(title_q, case=False, na=False)]

# Year range
if "Year" in df.columns and df["Year"].notna().any():
    ylo, yhi = int(df["Year"].min()), int(df["Year"].max())
    if ylo < yhi:
        yr = st.sidebar.slider("Release year", ylo, yhi, (ylo, yhi))
        df = df[df["Year"].between(*yr) | df["Year"].isna()]

# Runtime range (minutes)
if "Runtime (mins)" in df.columns and df["Runtime (mins)"].notna().any():
    rlo, rhi = int(df["Runtime (mins)"].min()), int(df["Runtime (mins)"].max())
    if rlo < rhi:
        rt = st.sidebar.slider("Runtime (minutes)", rlo, rhi, (rlo, rhi))
        df = df[df["Runtime (mins)"].between(*rt) | df["Runtime (mins)"].isna()]

# IMDb rating range
if "IMDb Rating" in df.columns and df["IMDb Rating"].notna().any():
    ir = st.sidebar.slider("IMDb rating", 0.0, 10.0,
        (float(round(df["IMDb Rating"].min(),1)), 10.0), 0.1)
    df = df[df["IMDb Rating"].between(*ir) | df["IMDb Rating"].isna()]

# Your rating range
if "Your Rating" in df.columns and df["Your Rating"].notna().any():
    yrr = st.sidebar.slider("Your rating", 1, 10,
        (int(df["Your Rating"].min()), 10))
    df = df[df["Your Rating"].between(*yrr) | df["Your Rating"].isna()]

# Votes minimum
if "Num Votes" in df.columns and df["Num Votes"].notna().any():
    vmax = int(df["Num Votes"].max())
    vmin = st.sidebar.slider("Minimum IMDb votes", 0, vmax, 0, step=max(1, vmax//100))
    df = df[df["Num Votes"].fillna(0) >= vmin]

# Genre filter (multiselect)
if "Genres" in df.columns:
    genre_opts = sorted(split_multi(df["Genres"]).unique().tolist())
    picked_genres = st.sidebar.multiselect("Genre(s)", genre_opts,
        help="Leave empty for all genres.")
    if picked_genres:
        pat = "|".join(picked_genres)
        df = df[df["Genres"].astype(str).str.contains(pat, case=False, na=False)]

# Director filter (multiselect)
if "Directors" in df.columns:
    dir_opts = (split_multi(df["Directors"]).value_counts().index.tolist())
    picked_dirs = st.sidebar.multiselect("Director(s)", dir_opts,
        help="Leave empty for all directors.")
    if picked_dirs:
        pat = "|".join(picked_dirs)
        df = df[df["Directors"].astype(str).str.contains(pat, case=False, na=False)]

if df.empty:
    st.warning("No ratings left after filters. Loosen the filters in the sidebar.")
    st.stop()

# ── Wrapped-style dashboard ───────────────────────────────────────────────────
n_films   = len(df)
avg_you   = df["Your Rating"].mean() if "Your Rating" in df else float("nan")
avg_imdb  = df["IMDb Rating"].mean() if "IMDb Rating" in df else float("nan")
total_min = df["Runtime (mins)"].sum() if "Runtime (mins)" in df else 0
total_hrs = total_min / 60
total_days = total_hrs / 24

top_genre = (split_multi(df["Genres"]).value_counts().idxmax()
             if "Genres" in df and df["Genres"].notna().any() else "—")
top_dir   = (split_multi(df["Directors"]).value_counts().idxmax()
             if "Directors" in df and df["Directors"].notna().any() else "—")

# favourite decade by count
fav_decade = "—"
if "Year" in df and df["Year"].notna().any():
    dec = ((df["Year"].dropna() // 10) * 10).astype(int)
    fav_decade = f"{dec.value_counts().idxmax()}s"

# generosity vs crowd
gen_txt = "—"
if pd.notna(avg_you) and pd.notna(avg_imdb):
    delta = avg_you - avg_imdb
    gen_txt = (f"+{delta:.1f} vs crowd (generous)" if delta >= 0
               else f"{delta:.1f} vs crowd (harsh)")

st.markdown(f"""
<div class="dash">
  <h2>MY IMDb RATINGS</h2>
  <div class="head">{n_films:,} FILMS RATED · {total_hrs:,.0f} HOURS OF FILM</div>
  <div class="row3">
    <div><div class="big">{avg_you:.1f}</div><div class="lbl">YOUR AVG</div></div>
    <div><div class="big">{avg_imdb:.1f}</div><div class="lbl">IMDb AVG</div></div>
    <div><div class="big">{total_days:,.0f}</div><div class="lbl">DAYS OF FILM</div></div>
  </div>
  <hr>
  <div class="row3" style="text-align:left;">
    <div>
      <div class="stat-lbl">FAVOURITE GENRE</div><div class="stat-val">{top_genre}</div>
      <div class="stat-lbl">MOST-RATED DIRECTOR</div><div class="stat-val">{top_dir}</div>
    </div>
    <div>
      <div class="stat-lbl">FAVOURITE DECADE</div><div class="stat-val">{fav_decade}</div>
      <div class="stat-lbl">YOUR GENEROSITY</div><div class="stat-val">{gen_txt}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.caption(f"{n_films:,} films analysed · Data: your IMDb ratings export"
           + (" · Movies only" if movies_only else ""))

def dark(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#000",
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs(
    ["Your ratings", "Genres", "Directors", "You vs IMDb", "Over time", "Discover"])

with t1:
    st.subheader("How you rate")
    c = df["Your Rating"].value_counts().sort_index().reset_index()
    c.columns = ["rating", "films"]
    st.plotly_chart(dark(px.bar(c, x="rating", y="films",
        color_discrete_sequence=[GOLD], labels={"rating":"Your rating","films":"Films"})
        .update_layout(xaxis=dict(dtick=1))), use_container_width=True)

    st.subheader("Highest-rated by you")
    cols = [c for c in ["Title","Year","Your Rating","IMDb Rating","Genres","Directors"]
            if c in df.columns]
    tbl = df.sort_values("Your Rating", ascending=False)[cols].head(50)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

with t2:
    st.subheader("Genres — count and how you rate them")
    if "Genres" in df.columns:
        g = df.dropna(subset=["Genres"]).copy()
        g["Genre"] = g["Genres"].astype(str).str.split(",")
        g = g.explode("Genre")
        g["Genre"] = g["Genre"].str.strip()
        g = g[g["Genre"] != ""]
        min_f = st.slider("Minimum films in genre", 1, 30, 3, key="gmin")
        stats = (g.groupby("Genre")
                   .agg(films=("Your Rating","size"), avg=("Your Rating","mean"))
                   .reset_index())
        stats = stats[stats["films"] >= min_f].sort_values("avg", ascending=False)
        if stats.empty:
            st.info("No genres meet that threshold — lower the minimum.")
        else:
            a, b = st.columns(2)
            a.plotly_chart(dark(px.bar(stats.sort_values("films"),
                x="films", y="Genre", orientation="h", title="Films watched",
                color_discrete_sequence=[GOLD])).update_layout(height=520),
                use_container_width=True)
            b.plotly_chart(dark(px.bar(stats.sort_values("avg"),
                x="avg", y="Genre", orientation="h", title="Your average rating",
                color_discrete_sequence=[TEAL])).update_layout(
                height=520, xaxis_range=[stats["avg"].min()-0.5, 10]),
                use_container_width=True)

with t3:
    st.subheader("Directors you rate most")
    if "Directors" in df.columns:
        d = df.dropna(subset=["Directors"]).copy()
        d["Director"] = d["Directors"].astype(str).str.split(",")
        d = d.explode("Director")
        d["Director"] = d["Director"].str.strip()
        d = d[d["Director"] != ""]
        min_f = st.slider("Minimum films by director", 1, 15, 2, key="dmin")
        stats = (d.groupby("Director")
                   .agg(films=("Your Rating","size"), avg=("Your Rating","mean"))
                   .reset_index())
        stats = stats[stats["films"] >= min_f]
        top_n = st.slider("How many", 10, 60, 25, key="dtop")
        by_count = stats.sort_values("films", ascending=False).head(top_n)
        st.plotly_chart(dark(px.bar(by_count.sort_values("films"),
            x="films", y="Director", orientation="h",
            color="avg", color_continuous_scale="YlOrBr",
            labels={"films":"Films rated","avg":"Your avg"})
            ).update_layout(height=max(400, len(by_count)*22)),
            use_container_width=True)
        st.caption("Bar length = films you've rated by them. Colour = your average rating for those films.")

with t4:
    st.subheader("Your rating vs the IMDb crowd")
    if {"Your Rating","IMDb Rating"}.issubset(df.columns):
        sc = df.dropna(subset=["Your Rating","IMDb Rating"]).copy()
        sc["delta"] = sc["Your Rating"] - sc["IMDb Rating"]
        hover = [c for c in ["Title","Year"] if c in sc.columns]
        fig = px.scatter(sc, x="IMDb Rating", y="Your Rating",
            hover_data=hover, color="delta", color_continuous_scale="RdYlGn",
            range_color=[-4, 4])
        fig.add_shape(type="line", x0=0, y0=0, x1=10, y1=10,
                      line=dict(color="#888", dash="dash"))
        fig.update_layout(xaxis_range=[0,10.5], yaxis_range=[0,10.5])
        st.plotly_chart(dark(fig).update_layout(height=520), use_container_width=True)
        st.caption("Points above the dashed line: you rated it higher than the crowd. Below: lower.")

        col1, col2 = st.columns(2)
        col1.markdown("**You loved these more than the crowd did**")
        col1.dataframe(sc.sort_values("delta", ascending=False)
            [[c for c in ["Title","Your Rating","IMDb Rating","delta"] if c in sc.columns]]
            .head(12).round(1), use_container_width=True, hide_index=True)
        col2.markdown("**You rated these well below the crowd**")
        col2.dataframe(sc.sort_values("delta")
            [[c for c in ["Title","Your Rating","IMDb Rating","delta"] if c in sc.columns]]
            .head(12).round(1), use_container_width=True, hide_index=True)

with t5:
    st.subheader("Your ratings over time")
    if "Date Rated" in df.columns and df["Date Rated"].notna().any():
        m = (df.dropna(subset=["Date Rated"]).set_index("Date Rated")
               .resample("MS")["Your Rating"].agg(["size","mean"]).reset_index())
        m.columns = ["month","films","avg"]
        a, b = st.columns(2)
        a.plotly_chart(dark(px.bar(m, x="month", y="films", title="Films rated per month",
            color_discrete_sequence=[GOLD], labels={"month":"","films":"Films"})),
            use_container_width=True)
        b.plotly_chart(dark(px.line(m, x="month", y="avg", title="Average rating given",
            markers=True, color_discrete_sequence=[TEAL], labels={"month":"","avg":"Avg rating"})),
            use_container_width=True)

    st.subheader("Taste by release decade")
    if "Year" in df.columns and df["Year"].notna().any():
        dd = df.dropna(subset=["Year"]).copy()
        dd["Decade"] = ((dd["Year"] // 10) * 10).astype(int).astype(str) + "s"
        agg = (dd.groupby("Decade")
                 .agg(films=("Your Rating","size"),
                      your_avg=("Your Rating","mean"),
                      imdb_avg=("IMDb Rating","mean"))
                 .reset_index().sort_values("Decade"))
        melt = agg.melt(id_vars="Decade", value_vars=["your_avg","imdb_avg"],
                        var_name="who", value_name="avg")
        melt["who"] = melt["who"].map({"your_avg":"You","imdb_avg":"IMDb crowd"})
        st.plotly_chart(dark(px.bar(melt, x="Decade", y="avg", color="who",
            barmode="group", color_discrete_sequence=[GOLD, TEAL],
            labels={"avg":"Average rating","who":""})), use_container_width=True)

with t6:
    st.subheader("Discover — great films you haven't rated")
    if catalog.empty:
        st.info("The reference catalog couldn't be loaded from GitHub, so Discover is unavailable. "
                "Check that imdb_movies.csv exists in your repo.")
    elif "Const" not in df.columns or "movie_id" not in catalog.columns:
        st.info("Can't match your ratings to the catalog (missing ID columns).")
    else:
        seen = set(df["Const"].dropna().astype(str))
        unseen = catalog[~catalog["movie_id"].astype(str).isin(seen)].copy()

        st.caption("Filters below apply to the catalog (imdb_movies.csv).")
        f1, f2, f3 = st.columns(3)

        # Title
        with f1:
            dtitle = st.text_input("Title contains", key="disc_title")
        if dtitle and "title" in unseen.columns:
            unseen = unseen[unseen["title"].astype(str).str.contains(dtitle, case=False, na=False)]

        # Genre + Director multiselects
        with f2:
            if "genre" in unseen.columns:
                gopts = sorted(split_multi(unseen["genre"]).unique().tolist())
                gpick = st.multiselect("Genre(s)", gopts, key="disc_g")
                if gpick:
                    pat = "|".join(gpick)
                    unseen = unseen[unseen["genre"].astype(str).str.contains(pat, case=False, na=False)]
        with f3:
            if "director" in unseen.columns:
                dopts = split_multi(unseen["director"]).value_counts().index.tolist()
                dpick = st.multiselect("Director(s)", dopts, key="disc_d")
                if dpick:
                    pat = "|".join(dpick)
                    unseen = unseen[unseen["director"].astype(str).str.contains(pat, case=False, na=False)]

        r1, r2, r3 = st.columns(3)
        # Year range
        with r1:
            if "year" in unseen.columns and unseen["year"].notna().any():
                ylo, yhi = int(unseen["year"].min()), int(unseen["year"].max())
                if ylo < yhi:
                    dyr = st.slider("Year", ylo, yhi, (ylo, yhi), key="disc_year")
                    unseen = unseen[unseen["year"].between(*dyr)]
        # Runtime range
        with r2:
            if "runtime" in unseen.columns and unseen["runtime"].notna().any():
                rlo, rhi = int(unseen["runtime"].min()), int(unseen["runtime"].max())
                if rlo < rhi:
                    drt = st.slider("Runtime (mins)", rlo, rhi, (rlo, rhi), key="disc_rt")
                    unseen = unseen[unseen["runtime"].between(*drt) | unseen["runtime"].isna()]
        # Rating range
        with r3:
            if "rating" in unseen.columns:
                minr = st.slider("IMDb rating", 6.8, 10.0, (7.5, 10.0), 0.1, key="disc_r")
                unseen = unseen[unseen["rating"].between(*minr)]

        # Votes minimum
        if "votes" in unseen.columns and unseen["votes"].notna().any():
            vmax = int(unseen["votes"].max())
            vmin = st.slider("Minimum votes", 0, vmax, 0, step=max(1, vmax//100), key="disc_v")
            unseen = unseen[unseen["votes"].fillna(0) >= vmin]

        st.write(f"🍿 {len(unseen):,} films in the catalog you haven't rated yet.")
        show_cols = [c for c in ["title","year","rating","votes","genre","director","movie_url"]
                     if c in unseen.columns]
        sort_col = "rating" if "rating" in unseen.columns else show_cols[0]
        st.dataframe(
            unseen.sort_values(sort_col, ascending=False)[show_cols].head(100),
            use_container_width=True, hide_index=True,
            column_config={"movie_url": st.column_config.LinkColumn("IMDb")}
                if "movie_url" in show_cols else None,
        )

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("Your rated films")
q = st.text_input("Search title, genre or director")
cols = [c for c in ["Title","Year","Your Rating","IMDb Rating","Runtime (mins)",
                    "Genres","Directors","Date Rated"] if c in df.columns]
out = df[cols].sort_values("Your Rating", ascending=False)
if q:
    mask = False
    for c in ["Title","Genres","Directors"]:
        if c in out.columns:
            mask = mask | out[c].astype(str).str.contains(q, case=False, na=False)
    out = out[mask]
st.dataframe(out, use_container_width=True, hide_index=True)
st.download_button("Download CSV", out.to_csv(index=False), "my_rated_films.csv", "text/csv")
