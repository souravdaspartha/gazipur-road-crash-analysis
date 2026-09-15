"""
ACCIDENT HOTSPOT MAP - GAZIPUR

Three layers:
    1. Crash density          (heatmap)
    2. Accident severity      (points coloured by severity)
    3. Top 5 accident locations

Run this inside your project folder:
    D:\\Personal\\GitHub\\Accident Hotspot
"""

import os
import warnings

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, HeatMapWithTime
from sklearn.cluster import DBSCAN

# ============================================================
# SETTINGS - change these, not the code below
# ============================================================
FOLDER   = r"D:\Personal\GitHub\Accident Hotspot"
CSV_FILE = "Road_Crash_Data.csv"

RADIUS   = 20     # size of each heat blob. Bigger = smoother, more merged
BLUR     = 18     # softness of the edges. Higher = no hard rim
MIN_OP   = 0.30   # strength of the weakest areas. High values cause a hard rim

# --- study area: crashes outside this are dropped ---
CRASH_MAX_LAT = 24.10   # cuts the Mawna and Bhaluka outliers
CRASH_MIN_LAT = None
CRASH_MAX_LON = None
CRASH_MIN_LON = None

# Years with too few records to be meaningful
EXCLUDE_YEARS = [2019]
MIN_YEAR = None    # or set a cutoff instead, e.g. 2020
MAX_YEAR = None

# Or remove individual points: (latitude, longitude, radius_km)
EXCLUDE_POINTS = [
    # (24.3600, 90.4000, 3.0),   # near Bhaluka
    # (24.2000, 90.4000, 3.0),   # near Mawna
]

EPS_KM   = 0.5    # cluster distance in kilometres (0.5 = 500 m)
MIN_PTS  = 5      # minimum crashes to count as an accident location

# --- dates ---
# Leave as None and the script works out the format itself and reports it.
# Or set it directly, e.g.
#   "%d/%m/%Y"  ->  05/03/2022      "%d-%m-%Y"  ->  05-03-2022
#   "%Y-%m-%d"  ->  2022-03-05      "%d %B %Y"  ->  5 March 2022
DATE_FORMAT = None

# --- labels ---
LABEL_COL = "Road Name"          # column holding the location name

# Your own names for the top locations, keyed by crash count.
# If two locations share a count, give a list ordered WEST to EAST.
NAME_OVERRIDES = {
    33: "Board Bazar Area",
    23: "Vogra Area",
    21: "Tongi Area",
    20: ["Konabari Area", "Salna Area"],   # tied count, west first
}
SHOW_SEVERITY_BY_DEFAULT = False # True = severity points on when map opens
SHOW_HOTSPOT_CIRCLES = True      # False = labels only, no dashed circle
LABEL_GAP_PX = 18                # gap between circle and its label

# --- theme ---
DARK_MODE = True          # dark basemap with glowing roads
SHOW_MAJOR_ROADS = True   # needs osmnx:  pip install osmnx
ROAD_CACHE = "major_roads.geojson"   # downloaded once, reused after

# Trim the road layer. Applied when drawing, so no re-download needed.
ROAD_PAD_DEG   = 0.02    # margin kept around the crash extent (~2 km)
ROAD_MAX_LAT   = None    # the boundary clip handles this now
ROAD_MIN_LAT   = None
ROAD_MAX_LON   = None
ROAD_MIN_LON   = None
# --- your own data (shapefile or GeoJSON) ---
# Put the files in the project folder. For a shapefile, keep the
# .shx .dbf .prj sidecar files next to the .shp.
LOCAL_ROADS_FILE = "gcc-road.shp"   # your roads, used instead of OSM
ROAD_CLASS_COL   = None   # not needed - all roads drawn the same
ROAD_CODE_COL    = None   # set this only if you want code badges

BOUNDARY_FILE  = "GCC_BND.shp"
CLIP_TO_BOUNDARY = True   # trim roads to the boundary
SHOW_BOUNDARY    = True   # draw the boundary outline as a layer
BOUNDARY_STYLE   = {"color": "#7de3ff", "weight": 2, "dash": "5,5",
                    "fill_opacity": 0.03}

# Roads counted as "crash roads" if a crash falls within this distance
ONLY_CODED_ROADS = False  # your file is already major roads only
SHOW_ROAD_CODES  = False  # needs ROAD_CODE_COL set
SHOW_CRASH_ROADS = False  # separate layer for roads carrying crashes

CRASH_ROAD_BUFFER_M = 50
CRASH_ROAD_STYLE = {"color": "#ff2d55", "core": "#ffe0e6", "w": (14, 7, 2.2)}

# --- colours ---
# Crash density: light green (low) -> yellow -> deep red (high)
GRADIENT = {
    0.03: "#c7e9b4",
    0.18: "#ffeda0",
    0.38: "#fed976",
    0.58: "#feb24c",
    0.75: "#fd8d3c",
    0.90: "#f03b20",
    1.00: "#bd0026",
}
GRADIENT_CSS = "#c7e9b4,#ffeda0,#fed976,#feb24c,#fd8d3c,#f03b20,#bd0026"

ACCENT      = "#ffffff" if DARK_MODE else "#1a1a1a"   # hotspot circles/labels
TILES       = "CartoDB dark_matter" if DARK_MODE else "CartoDB positron"
DOT_OUTLINE = "#ffffff" if DARK_MODE else "#1a1a1a"
LABEL_SUB   = "#c8d6e0" if DARK_MODE else "#555555"
LEG_BG      = "rgba(18,20,24,.88)" if DARK_MODE else "rgba(255,255,255,.93)"
LEG_FG      = "#e8eef3" if DARK_MODE else "#444444"
LEG_BORDER  = "#3a4048" if DARK_MODE else "#cccccc"
HALO        = "#000000" if DARK_MODE else "#ffffff"

# All roads share one colour; hierarchy is shown by line width instead
ROAD_COLOR = "#6b8a9c"   # the glow - cool grey-blue
ROAD_CORE  = "#eaf2f7"   # the bright centre line - near white

ROAD_WIDTHS = (5, 3, 0.9)   # (outer glow, mid band, bright core) in pixels

ROAD_STYLE = {"color": ROAD_COLOR, "core": ROAD_CORE, "w": ROAD_WIDTHS}

# Severity: cool family, so it stays distinct from the density ramp
SEVERITY_STYLE = {
    "Fatal":         {"color": "#e91e63", "size": 7},   # crimson pink
    "Severe Injury": {"color": "#7e57c2", "size": 6},   # purple
    "Minor Injury":  {"color": "#4fc3f7", "size": 5},   # light blue
    "Collision":     {"color": "#b2ebf2", "size": 4},   # pale cyan
}
DEFAULT_STYLE = {"color": "#adb5bd", "size": 4}

# --- animated map ---
ANIM_FILE   = "accident_hotspot_animated.html"   # full interactive version
EMBED_FILE  = "accident_hotspot_embed.html"      # locked, for the website
EMBED_ZOOM  = 11.95   # fractional allowed. Lower = more space around the data
CENTRE_ON_BOUNDARY = True  # centre on GCC_BND rather than the crash points
EMBED_SHIFT_LAT = 0.0      # negative moves the data UP in the frame
EMBED_SHIFT_LON = 0.0      # POSITIVE moves the data LEFT in the frame
SHOW_NORTH_ARROW = True
ARROW_SIZE = 48            # north arrow size in pixels
COMPACT_TIME_CONTROL = True   # tighten the year cell and sliders
DATE_CELL_PX  = 54            # width of the year box
SLIDER_PX     = 130           # width of the year slider
HIDE_SPEED    = False         # True removes the fps control

ANIM_RADIUS = 28
ANIM_SPEED  = 0.6

os.chdir(FOLDER)

# ============================================================
# 1. LOAD AND CLEAN
# ============================================================
df = pd.read_csv(CSV_FILE, encoding="latin-1")

# strip stray spaces from column names -> 'Latitude ' becomes 'Latitude'
df.columns = df.columns.str.strip()
print("Columns found:", df.columns.tolist())

# drop any 'Total' summary row
df = df[df["Date"].astype(str).str.strip().str.lower() != "total"]

def parse_dates(series, fmt=None):
    """Parse dates without the 'could not infer format' warning.

    If fmt is given, use it. Otherwise try common patterns and keep
    whichever one reads the most rows.
    """
    if fmt:
        return pd.to_datetime(series, format=fmt, errors="coerce"), fmt

    candidates = [
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y",
        "%d/%m/%y", "%d-%b-%Y", "%d %B %Y", "%d %b %Y",
        "%Y/%m/%d", "%d.%m.%Y",
    ]
    best, best_fmt, best_hits = None, None, -1
    for c in candidates:
        try:
            out = pd.to_datetime(series, format=c, errors="coerce")
        except (ValueError, TypeError):
            continue
        hits = int(out.notna().sum())
        if hits > best_hits:
            best, best_fmt, best_hits = out, c, hits

    # if no single pattern reads nearly everything, fall back to per-row
    if best_hits < len(series.dropna()) * 0.95:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out = pd.to_datetime(series, format="mixed", dayfirst=True,
                                     errors="coerce")
            except (ValueError, TypeError):
                out = pd.to_datetime(series, dayfirst=True, errors="coerce")
        return out, "mixed"

    return best, best_fmt


df["Date"], used_fmt = parse_dates(df["Date"], DATE_FORMAT)
print(f"Date format used: {used_fmt}")

bad_dates = int(df["Date"].isna().sum())
if bad_dates:
    print(f"WARNING: {bad_dates} row(s) had an unreadable date and were dropped.")
df = df.dropna(subset=["Date"])

# coordinates must be real numbers
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df = df.dropna(subset=["Latitude", "Longitude"])

# sanity check - anything outside Bangladesh is a typo
before = len(df)
df = df[df["Latitude"].between(20.5, 26.7) & df["Longitude"].between(88.0, 92.7)]
if len(df) < before:
    print(f"Dropped {before - len(df)} record(s) with impossible coordinates")

# study area filter
for name, col, op in (
    ("CRASH_MAX_LAT", "Latitude", "max"), ("CRASH_MIN_LAT", "Latitude", "min"),
    ("CRASH_MAX_LON", "Longitude", "max"), ("CRASH_MIN_LON", "Longitude", "min"),
):
    limit = globals()[name]
    if limit is None:
        continue
    n0 = len(df)
    df = df[df[col] <= limit] if op == "max" else df[df[col] >= limit]
    if len(df) < n0:
        print(f"{name}={limit}: dropped {n0 - len(df)} crash(es)")

# remove individual points by coordinate
for plat, plon, prad in EXCLUDE_POINTS:
    d = np.degrees(2 * np.arcsin(np.sqrt(
        np.sin(np.radians(df["Latitude"] - plat) / 2) ** 2
        + np.cos(np.radians(plat)) * np.cos(np.radians(df["Latitude"]))
        * np.sin(np.radians(df["Longitude"] - plon) / 2) ** 2
    ))) * 111.32
    n0 = len(df)
    df = df[d > prad]
    if len(df) < n0:
        print(f"Excluded {n0 - len(df)} crash(es) within {prad} km "
              f"of ({plat}, {plon})")

# severity weighting - fatal crashes push the heat up harder
weights = {"Fatal": 5, "Severe Injury": 3, "Minor Injury": 1, "Collision": 1}
df["weight"] = df["Accident Severity"].map(weights).fillna(1)

# location name used for labels
if LABEL_COL not in df.columns:
    print(f"WARNING: '{LABEL_COL}' not found. Labels will say 'Unknown'.")
    print(f"Pick one of these instead: {df.columns.tolist()}")
    df["_label"] = "Unknown"
else:
    df["_label"] = df[LABEL_COL].fillna("Unknown").astype(str).str.strip()
    df.loc[df["_label"] == "", "_label"] = "Unknown"

df["Year"] = df["Date"].dt.year

# year filter
year_counts = df["Year"].value_counts().sort_index()
print("Records per year before filtering:")
for y, c in year_counts.items():
    print(f"  {int(y)}: {c}")

if EXCLUDE_YEARS:
    n0 = len(df)
    df = df[~df["Year"].isin(EXCLUDE_YEARS)]
    if len(df) < n0:
        print(f"Excluded {EXCLUDE_YEARS}: dropped {n0 - len(df)} crash(es)")

for name, op in (("MIN_YEAR", "min"), ("MAX_YEAR", "max")):
    limit = globals()[name]
    if limit is None:
        continue
    n0 = len(df)
    df = df[df["Year"] >= limit] if op == "min" else df[df["Year"] <= limit]
    if len(df) < n0:
        print(f"{name}={limit}: dropped {n0 - len(df)} crash(es)")

print(f"Crashes mapped: {len(df)}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ============================================================
# 2. MAJOR ROADS  (downloaded from OpenStreetMap, cached locally)
# ============================================================
def get_major_roads(bounds, pad=0.02):
    """Return a GeoDataFrame of major roads, or None if unavailable.

    Downloads once and caches to ROAD_CACHE, so later runs are instant.
    """
    try:
        import geopandas as gpd
    except ImportError:
        print("SKIP roads: geopandas not installed  (pip install osmnx)")
        return None

    # your own road file takes priority over OpenStreetMap
    if LOCAL_ROADS_FILE:
        if not os.path.exists(LOCAL_ROADS_FILE):
            print(f"SKIP roads: {LOCAL_ROADS_FILE} not found in {FOLDER}")
            return None
        edges = gpd.read_file(LOCAL_ROADS_FILE)
        print(f"Roads loaded from {LOCAL_ROADS_FILE}: {len(edges)} features")
        print(f"  columns: {[c for c in edges.columns if c != 'geometry']}")

        if edges.crs is None:
            print("  WARNING: no CRS found, assuming WGS84 (EPSG:4326)")
            edges = edges.set_crs("EPSG:4326")
        elif edges.crs.to_epsg() != 4326:
            edges = edges.to_crs("EPSG:4326")
            print(f"  reprojected to EPSG:4326")

        # optional code column for the badges
        if ROAD_CODE_COL and ROAD_CODE_COL in edges.columns:
            edges["ref"] = edges[ROAD_CODE_COL].astype(str)
        else:
            edges["ref"] = None
            if ROAD_CODE_COL:
                print(f"  note: '{ROAD_CODE_COL}' not found, no code badges")

        return edges[["ref", "geometry"]]

    if os.path.exists(ROAD_CACHE):
        print(f"Roads loaded from cache: {ROAD_CACHE}")
        return gpd.read_file(ROAD_CACHE)

    try:
        import osmnx as ox
    except ImportError:
        print("SKIP roads: osmnx not installed  (pip install osmnx)")
        return None

    south, north, west, east = bounds
    south, north = south - pad, north + pad
    west, east = west - pad, east + pad
    cf = '["highway"~"motorway|trunk|primary|secondary"]'

    print("Downloading roads from OpenStreetMap (one time, ~1 min)...")
    try:
        try:   # osmnx 2.x
            G = ox.graph_from_bbox(bbox=(west, south, east, north),
                                   custom_filter=cf, retain_all=True)
        except TypeError:   # osmnx 1.x
            G = ox.graph_from_bbox(north=north, south=south, east=east,
                                   west=west, custom_filter=cf,
                                   retain_all=True)
        edges = ox.graph_to_gdfs(G, nodes=False)
    except Exception as e:
        print(f"SKIP roads: download failed ({type(e).__name__}: {e})")
        return None

    edges = edges.reset_index()
    for col in ("ref", "name"):
        if col not in edges.columns:
            edges[col] = None
    edges = edges[["highway", "ref", "name", "geometry"]]

    # OSM tags are sometimes lists when a way carries several values
    def first(v):
        return v[0] if isinstance(v, list) else v

    for col in ("highway", "ref", "name"):
        edges[col] = edges[col].apply(first)
    edges.to_file(ROAD_CACHE, driver="GeoJSON")
    print(f"Roads cached to {ROAD_CACHE} ({len(edges)} segments)")
    return edges


def clip_roads(edges, bounds):
    """Trim roads to the study area before drawing."""
    from shapely.geometry import box

    if BOUNDARY_FILE and CLIP_TO_BOUNDARY and os.path.exists(BOUNDARY_FILE):
        import geopandas as gpd
        poly = gpd.read_file(BOUNDARY_FILE).to_crs(edges.crs)
        out = edges.clip(poly)
        print(f"Roads clipped to {BOUNDARY_FILE}: {len(out)} segments kept")
        return out

    south, north, west, east = bounds
    south = ROAD_MIN_LAT if ROAD_MIN_LAT is not None else south - ROAD_PAD_DEG
    north = ROAD_MAX_LAT if ROAD_MAX_LAT is not None else north + ROAD_PAD_DEG
    west  = ROAD_MIN_LON if ROAD_MIN_LON is not None else west - ROAD_PAD_DEG
    east  = ROAD_MAX_LON if ROAD_MAX_LON is not None else east + ROAD_PAD_DEG

    before = len(edges)
    out = edges.clip(box(west, south, east, north))
    print(f"Roads clipped to [{south:.3f}-{north:.3f} N, "
          f"{west:.3f}-{east:.3f} E]: {len(out)} of {before} segments kept")
    return out


def _draw(pts, fg, style, boost=0.0):
    """Three passes: wide faint halo, mid band, bright core."""
    w_out, w_mid, w_core = style["w"]
    for width, colour, op in (
        (w_out,  style["color"], 0.10 + boost),
        (w_mid,  style["color"], 0.28 + boost),
        (w_core, style["core"],  0.85),
    ):
        folium.PolyLine(pts, color=colour, weight=width,
                        opacity=min(op, 1.0), smooth_factor=1.5).add_to(fg)


def _coords(geom):
    """Every line in a geometry, as (lat, lon) lists."""
    geoms = (list(geom.geoms) if geom.geom_type == "MultiLineString"
             else [geom])
    return [[(y, x) for x, y in g.coords] for g in geoms]


def keep_coded(edges):
    """Drop roads with no reference code (N3, R301, Z3007 ...)."""
    if "ref" not in edges.columns:
        print("SKIP code filter: no 'ref' column - "
              "delete major_roads.geojson and rerun to re-download")
        return edges

    before = len(edges)
    out = edges[edges["ref"].notna() & (edges["ref"].astype(str) != "")]
    codes = sorted(out["ref"].astype(str).unique())
    print(f"Coded roads kept: {len(out)} of {before} segments")
    print(f"  codes found: {', '.join(codes)}")
    return out


def add_road_layers(edges, target_map, show=True):
    """All major roads in a single toggleable layer, codes included."""
    fg = folium.FeatureGroup(name="Major roads", show=show)

    n = 0
    for _, e in edges.iterrows():
        if e.geometry is None:
            continue
        for pts in _coords(e.geometry):
            _draw(pts, fg, ROAD_STYLE)
            n += 1

    if SHOW_ROAD_CODES and "ref" in edges.columns:
        coded = edges[edges["ref"].notna()].copy()
        if len(coded):
            coded["_len"] = coded.geometry.length
            for code, grp in coded.groupby(coded["ref"].astype(str)):
                longest = grp.loc[grp["_len"].idxmax()].geometry
                lines = _coords(longest)
                if not lines:
                    continue
                pts = max(lines, key=len)
                lat, lon = pts[len(pts) // 2]
                colour = ROAD_CORE
                html = (
                    f'<div style="font-family: Arial, sans-serif;'
                    f' font-size: 11px; font-weight: bold; color: {colour};'
                    f' text-align: center; white-space: nowrap;'
                    f' padding: 1px 5px; border-radius: 3px;'
                    f' background: rgba(0,0,0,.55);'
                    f' text-shadow: 0 0 4px {HALO};">{code}</div>'
                )
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.DivIcon(icon_size=(60, 18),
                                        icon_anchor=(30, 9), html=html),
                ).add_to(fg)

    fg.add_to(target_map)
    print(f"  Major roads: {n} segments")


_boundary = None
_boundary_loaded = False


def load_boundary():
    """Read BOUNDARY_FILE once, in WGS84. Returns None if unavailable."""
    global _boundary, _boundary_loaded
    if _boundary_loaded:
        return _boundary

    _boundary_loaded = True
    _boundary = None
    if not BOUNDARY_FILE:
        return None
    if not os.path.exists(BOUNDARY_FILE):
        print(f"SKIP boundary: {BOUNDARY_FILE} not found in {FOLDER}")
        return None

    try:
        import geopandas as gpd
    except ImportError:
        print("SKIP boundary: geopandas not installed")
        return None

    poly = gpd.read_file(BOUNDARY_FILE)
    if poly.crs is None:
        poly = poly.set_crs("EPSG:4326")
    elif poly.crs.to_epsg() != 4326:
        poly = poly.to_crs("EPSG:4326")

    _boundary = poly
    return poly


def add_boundary(target_map, show=True):
    """Draw the study area outline from BOUNDARY_FILE."""
    if not SHOW_BOUNDARY:
        return
    poly = load_boundary()
    if poly is None:
        return

    st = BOUNDARY_STYLE
    folium.GeoJson(
        poly.__geo_interface__,
        name="Study area",
        show=show,
        style_function=lambda _: {
            "color": st["color"],
            "weight": st["weight"],
            "dashArray": st["dash"],
            "fillColor": st["color"],
            "fillOpacity": st["fill_opacity"],
        },
    ).add_to(target_map)
    print(f"Boundary drawn from {BOUNDARY_FILE}: {len(poly)} feature(s)")


def roads_with_crashes(edges, crashes, dist_m=CRASH_ROAD_BUFFER_M):
    """Major roads with at least one crash within dist_m."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        return None

    # metres need a projected CRS - UTM zone 46N covers Bangladesh
    utm = 32646
    pts = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in
                  zip(crashes["Longitude"], crashes["Latitude"])],
        crs="EPSG:4326",
    ).to_crs(utm)

    proj = edges.to_crs(utm)
    buffered = pts.buffer(dist_m)
    try:
        blob = buffered.union_all()      # geopandas 1.x
    except AttributeError:
        blob = buffered.unary_union      # older versions

    hit = proj.intersects(blob)
    out = edges[hit.values]
    print(f"Roads with crashes within {dist_m} m: "
          f"{len(out)} of {len(edges)} segments")
    return out


def add_crash_roads(edges, target_map, show=True):
    fg = folium.FeatureGroup(name="Roads with crashes", show=show)
    for _, e in edges.iterrows():
        if e.geometry is None:
            continue
        for pts in _coords(e.geometry):
            _draw(pts, fg, CRASH_ROAD_STYLE, boost=0.08)
    fg.add_to(target_map)


bounds = (df["Latitude"].min(), df["Latitude"].max(),
          df["Longitude"].min(), df["Longitude"].max())

road_edges = get_major_roads(bounds) if SHOW_MAJOR_ROADS else None
if road_edges is not None and len(road_edges):
    road_edges = clip_roads(road_edges, bounds)

print(f"Crash extent: {bounds[0]:.3f}-{bounds[1]:.3f} N, "
      f"{bounds[2]:.3f}-{bounds[3]:.3f} E")

if road_edges is not None and len(road_edges) and ONLY_CODED_ROADS:
    road_edges = keep_coded(road_edges)

crash_road_edges = (roads_with_crashes(road_edges, df)
                    if SHOW_CRASH_ROADS and road_edges is not None
                    and len(road_edges) else None)

# ============================================================
# 3. BASE MAP
# ============================================================
m = folium.Map(
    location=[df["Latitude"].mean(), df["Longitude"].mean()],
    zoom_start=12,
    tiles=None,
    control_scale=True,
)
folium.TileLayer(TILES, control=False).add_to(m)

# ============================================================
# 4. LAYER 1 - CRASH DENSITY
# ============================================================
add_boundary(m, show=True)
if road_edges is not None:
    print("Road layers:")
    add_road_layers(road_edges, m, show=True)
if SHOW_CRASH_ROADS and crash_road_edges is not None and len(crash_road_edges):
    add_crash_roads(crash_road_edges, m, show=False)

HeatMap(
    df[["Latitude", "Longitude", "weight"]].values.tolist(),
    name="Crash Density",
    radius=RADIUS,
    blur=BLUR,
    min_opacity=MIN_OP,
    gradient=GRADIENT,
).add_to(m)

# ============================================================
# 5. LAYER 2 - CRASH SEVERITY
# ============================================================
fg_sev = folium.FeatureGroup(
    name="Crash severity",
    show=SHOW_SEVERITY_BY_DEFAULT,
)


def popup_html(row):
    fields = [
        ("Accident No.", "Accident No."),
        ("Time", "Time"),
        ("Severity", "Accident Severity"),
        ("Collision", "Collision Type"),
        ("Road", "Road Name"),
        ("Casualties", "Total Casualties"),
    ]
    lines = [f"<b>Date:</b> {row['Date'].strftime('%d %b %Y')}"]
    for label, col in fields:
        if col in df.columns:
            lines.append(f"<b>{label}:</b> {row[col]}")
    return "<br>".join(lines)


# draw least severe first so fatal points sit on top
order = ["Collision", "Minor Injury", "Severe Injury", "Fatal"]
df["_sev_rank"] = df["Accident Severity"].apply(
    lambda v: order.index(v) if v in order else -1
)

for _, row in df.sort_values("_sev_rank").iterrows():
    style = SEVERITY_STYLE.get(row["Accident Severity"], DEFAULT_STYLE)
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=style["size"],
        color=DOT_OUTLINE,
        weight=1,
        fill=True,
        fill_color=style["color"],
        fill_opacity=0.9,
        popup=folium.Popup(popup_html(row), max_width=280),
        tooltip=folium.Tooltip(
            f"{row['Accident Severity']} - {row['_label']}",
            sticky=True,
        ),
    ).add_to(fg_sev)

fg_sev.add_to(m)

# ============================================================
# 6. LAYER 3 - TOP 5 ACCIDENT LOCATIONS  (DBSCAN)
# ============================================================
coords_rad = np.radians(df[["Latitude", "Longitude"]].values)
EARTH_KM = 6371.0

db = DBSCAN(
    eps=EPS_KM / EARTH_KM,
    min_samples=MIN_PTS,
    metric="haversine",
).fit(coords_rad)

df["cluster"] = db.labels_

counts = df[df["cluster"] != -1]["cluster"].value_counts()
top_ids = counts.head(5).index.tolist()

# build the summary table first
rows = []
for rank, cid in enumerate(top_ids, start=1):
    sub = df[df["cluster"] == cid]
    road = sub["_label"].mode()
    rows.append({
        "Rank": rank,
        "Location": road.iloc[0] if len(road) else "Unknown",
        "Latitude": round(sub["Latitude"].mean(), 6),
        "Longitude": round(sub["Longitude"].mean(), 6),
        "Total_Crashes": len(sub),
        "Fatal_Crashes": int((sub["Accident Severity"] == "Fatal").sum()),
    })


def apply_name_overrides(rows):
    """Replace auto-detected road names with the names in NAME_OVERRIDES.

    Matching is by crash count. Where several locations share a count,
    the names are assigned west to east by longitude.
    """
    groups = {}
    for r in rows:
        groups.setdefault(r["Total_Crashes"], []).append(r)

    for count, names in NAME_OVERRIDES.items():
        if count not in groups:
            print(f"  note: no location has {count} crashes, "
                  f"'{names}' not applied")
            continue

        if isinstance(names, str):
            names = [names]

        tied = sorted(groups[count], key=lambda r: r["Longitude"])

        if len(names) < len(tied):
            print(f"  note: {len(tied)} locations have {count} crashes "
                  f"but only {len(names)} name(s) given")

        for r, name in zip(tied, names):
            r["Location"] = name

    return rows


rows = apply_name_overrides(rows)

print("\nTop 5 accident locations:")
for r in rows:
    print(f"  #{r['Rank']}  {r['Location']:<20} "
          f"{r['Total_Crashes']} crashes ({r['Fatal_Crashes']} fatal)  "
          f"[{r['Latitude']:.4f}, {r['Longitude']:.4f}]")


def build_top_layer():
    """A fresh layer of the top 5 locations.

    Folium can't share one FeatureGroup between two maps, so this is
    called once per map.
    """
    fg = folium.FeatureGroup(name="Top 5 accident locations", show=True)

    for r in rows:
        lat, lon = r["Latitude"], r["Longitude"]

        if SHOW_HOTSPOT_CIRCLES:
            folium.Circle(
                location=[lat, lon],
                radius=400,
                color=ACCENT,
                weight=2,
                dash_array="6,6",
                fill=True,
                fill_color=ACCENT,
                fill_opacity=0.06,
                popup=(f"<b>Location #{r['Rank']}</b><br>{r['Location']}<br>"
                       f"Crashes: {r['Total_Crashes']}<br>"
                       f"Fatal: {r['Fatal_Crashes']}"),
                tooltip=(f"#{r['Rank']} {r['Location']} - "
                         f"{r['Total_Crashes']} crashes"),
            ).add_to(fg)

        # label sits to the RIGHT of the circle, vertically centred
        label_html = (
            f'<div style="font-family: Arial, sans-serif; font-size: 12px;'
            f' font-weight: bold; color: {ACCENT}; text-align: left;'
            f' line-height: 1.3; white-space: nowrap;'
            f' text-shadow: 0 0 4px {HALO}, 0 0 8px {HALO}, 0 0 2px {HALO};">'
            f'#{r["Rank"]} {r["Location"]}<br>'
            f'<span style="font-weight: normal; font-size: 11px;'
            f' color: {LABEL_SUB};">'
            f'{r["Total_Crashes"]} crashes</span></div>'
        )
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                icon_size=(200, 34),
                # negative x pushes the box right of the point
                icon_anchor=(-LABEL_GAP_PX, 17),
                html=label_html,
            ),
        ).add_to(fg)

    return fg


build_top_layer().add_to(m)

# ============================================================
# 7. LEGEND
# ============================================================
sev_rows = "".join(
    f'<div style="display:flex;align-items:center;margin-top:2px;">'
    f'<span style="width:8px;height:8px;border-radius:50%;'
    f'background:{v["color"]};border:1px solid {LEG_BORDER};'
    f'margin-right:5px;flex:none;"></span>{k}</div>'
    for k, v in SEVERITY_STYLE.items()
)

legend = f"""
<div style="position: fixed; bottom: 12px; left: 12px; z-index: 9999;
            background: {LEG_BG}; padding: 7px 9px;
            border: 1px solid {LEG_BORDER}; border-radius: 4px;
            font-family: Arial, sans-serif; font-size: 10px; color: {LEG_FG};
            line-height: 1.25; box-shadow: 0 1px 6px rgba(0,0,0,.4);">
  <div style="font-weight: bold; font-size: 10px;">Crash density</div>
  <div style="width: 104px; height: 8px; border-radius: 2px; margin-top: 3px;
       background: linear-gradient(to right, {GRADIENT_CSS});"></div>
  <div style="display: flex; justify-content: space-between; width: 104px;
              opacity: .7; font-size: 9px;">
    <span>Low</span><span>High</span>
  </div>
  <div style="font-weight: bold; margin-top: 7px;">Severity</div>
  {sev_rows}
</div>
"""
TIME_CSS = f"""
<style>
.leaflet-bar-timecontrol .timecontrol-date {{
  min-width: {DATE_CELL_PX}px !important;
  width: {DATE_CELL_PX}px !important;
  padding: 0 4px !important;
  font-size: 12px !important;
  text-align: center !important;
}}
.leaflet-bar-timecontrol .timecontrol-slider,
.leaflet-bar-timecontrol .timecontrol-dateslider {{
  width: {SLIDER_PX}px !important;
  min-width: {SLIDER_PX}px !important;
}}
.leaflet-bar-timecontrol a {{
  padding-left: 4px !important;
  padding-right: 4px !important;
}}
{'.leaflet-bar-timecontrol .timecontrol-speed { display: none !important; }'
 if HIDE_SPEED else ''}
</style>
"""

_a = ARROW_SIZE
NORTH_ARROW = f"""
<div style="position: fixed; top: 14px; right: 14px; z-index: 9999;
            width: {_a}px; text-align: center; font-family: Arial, sans-serif;">
  <svg width="{_a}" height="{_a}" viewBox="0 0 34 34">
    <polygon points="17,2 25,28 17,22 9,28"
             fill="{LEG_FG}" stroke="{LEG_FG}" stroke-width="1.2"
             stroke-linejoin="round"/>
    <polygon points="17,2 17,22 9,28" fill="{LEG_BG}"/>
  </svg>
  <div style="font-size: {int(_a * 0.34)}px; font-weight: bold; color: {LEG_FG};
              margin-top: -4px; text-shadow: 0 0 5px {HALO};">N</div>
</div>
"""

m.get_root().html.add_child(folium.Element(legend))
if SHOW_NORTH_ARROW:
    m.get_root().html.add_child(folium.Element(NORTH_ARROW))
folium.LayerControl(collapsed=False).add_to(m)

# ============================================================
# 8. ANIMATED MAP - year by year time slider
# ============================================================
years = sorted(df["Year"].dropna().unique().astype(int))
max_w = df["weight"].max()

frames = [
    [[r["Latitude"], r["Longitude"], r["weight"] / max_w]
     for _, r in df[df["Year"] == y].iterrows()]
    for y in years
]

def build_anim_map(locked=False):
    """Year-by-year map.

    locked=True removes zoom, dragging and the layer control, so the
    embedded version always opens on the same view.
    """
    # centre on the boundary if we have one, otherwise on the crash extent
    poly = load_boundary() if CENTRE_ON_BOUNDARY else None
    if poly is not None and len(poly):
        west, south, east, north = poly.total_bounds
    else:
        south, north = df["Latitude"].min(), df["Latitude"].max()
        west, east = df["Longitude"].min(), df["Longitude"].max()

    centre = [
        (south + north) / 2 + EMBED_SHIFT_LAT,
        (west + east) / 2 + EMBED_SHIFT_LON,
    ]

    opts = dict(
        location=centre,
        zoom_start=EMBED_ZOOM,
        tiles=None,
        control_scale=not locked,
    )
    if locked:
        opts.update(
            zoom_control=False,
            scrollWheelZoom=False,
            dragging=False,
            doubleClickZoom=False,
            touchZoom=False,
            boxZoom=False,
            keyboard=False,
            min_zoom=EMBED_ZOOM,   # pin the scale so nothing can change it
            max_zoom=EMBED_ZOOM,
            zoomSnap=0.1,          # allows fractional EMBED_ZOOM
            zoomDelta=0.1,
        )

    amap = folium.Map(**opts)
    folium.TileLayer(TILES, control=False).add_to(amap)

    HeatMapWithTime(
        frames,
        name="Crash Density",
        index=[str(y) for y in years],
        gradient={str(k): v for k, v in GRADIENT.items()},
        radius=ANIM_RADIUS,
        min_opacity=0.4,
        max_opacity=0.9,
        use_local_extrema=False,   # fixed colour scale across all years
        # locked map has no zoom buttons, so the top left is free
        position="topleft" if locked else "bottomright",
        auto_play=True,
        display_index=True,
        speed_step=ANIM_SPEED,
    ).add_to(amap)

    add_boundary(amap, show=True)
    if road_edges is not None:
        add_road_layers(road_edges, amap, show=True)
    if SHOW_CRASH_ROADS and crash_road_edges is not None and len(crash_road_edges):
        add_crash_roads(crash_road_edges, amap, show=False)

    build_top_layer().add_to(amap)
    amap.get_root().html.add_child(folium.Element(legend))
    if COMPACT_TIME_CONTROL:
        amap.get_root().html.add_child(folium.Element(TIME_CSS))
    if SHOW_NORTH_ARROW:
        amap.get_root().html.add_child(folium.Element(NORTH_ARROW))

    if not locked:
        # interactive version can frame itself to the data
        amap.fit_bounds([
            [df["Latitude"].min(), df["Longitude"].min()],
            [df["Latitude"].max(), df["Longitude"].max()],
        ])
        folium.LayerControl(collapsed=False).add_to(amap)

    return amap


_p = load_boundary() if CENTRE_ON_BOUNDARY else None
if _p is not None and len(_p):
    _w, _s2, _e, _n = _p.total_bounds
    print(f"Locked view centred on boundary: "
          f"{(_s2 + _n) / 2:.4f}, {(_w + _e) / 2:.4f} at zoom {EMBED_ZOOM}")
else:
    print(f"Locked view centred on crash extent: "
          f"{(df['Latitude'].min() + df['Latitude'].max()) / 2:.4f}, "
          f"{(df['Longitude'].min() + df['Longitude'].max()) / 2:.4f} "
          f"at zoom {EMBED_ZOOM}")

build_anim_map(locked=False).save(ANIM_FILE)
build_anim_map(locked=True).save(EMBED_FILE)

# ============================================================
# 9. SAVE
# ============================================================
m.save("accident_hotspot_map.html")
pd.DataFrame(rows).to_csv("top_5_hotspots.csv", index=False)
df[df["cluster"] != -1].to_csv("hotspot_clusters.csv", index=False)

print("\nCrashes per year:")
for y, f in zip(years, frames):
    print(f"  {y}: {len(f)}")

print("\nSaved:")
print("  accident_hotspot_map.html")
print(f"  {ANIM_FILE}   (interactive)")
print(f"  {EMBED_FILE}   (locked, for the website)")
print("  top_5_hotspots.csv")
print("  hotspot_clusters.csv")
print(f"\nClusters found: {len(counts)}  |  Crashes inside clusters: {int(counts.sum())}")
print(pd.DataFrame(rows).to_string(index=False))