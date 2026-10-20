"""Feature preprocessing: fit on training rows, then transform any split.

:func:`fit_preprocessor` receives training rows only, and :func:`transform`
reuses its learned decisions and statistics without fitting on validation or
test rows. This prevents preprocessing leakage.

Owner: data role (Ahmed).

Two families of options
-----------------------
v1 options (Peter, 25 Sep) learn everything from the data:
``one_hot``, ``poly_degree`` and ``missing_indicators``, plus a heuristic that
guesses "don't know / refused" codes such as 7/9.

v2 options (Ahmed, 4 Oct) apply what the CDC codebook says, as recorded per
column in ``docs/feature_registry.csv``. They are cumulative steps of one plan
(``docs/preprocessing_plan.pdf``), each measured on its own in
``experiments/261004_aa_preprocessing_v2.ipynb``:

=====  ==========================  ==============================================
step   option                      what it does
=====  ==========================  ==============================================
1      ``codes=True``              codebook codes -> NaN or 0, skip-logic fills,
                                   one unit per column (UNIT_RULES); replaces
                                   the sentinel heuristic
2      ``flags=True``              0/1 flags for the module questions HAREHAB1,
                                   CVDASPRN and RDUCHART (see below)
3      ``registry_one_hot=True``   one-hot exactly the columns the registry marks
                                   ``one-hot``, with an "unknown" level
4      ``nonlinear=True``          age^2, BMI^2; clip + log1p for heavy tails
5      ``extras=True``             _STATE one-hot, years with diabetes, one
                                   "adults in household" column
6      ``select=True``             keep exactly the registry's non-drop columns
                                   (a lean set; measured, costs a little F1)
7      ``engineered=True``         products and curves a linear model cannot
                                   build itself (ENGINEERED_DEFAULT, see below)
8      ``aspirin_interactions``    the aspirin-module answer x age, sex, general
                                   health and 4 risk factors (experimental, see
                                   ASPIRIN_FACTORS)
=====  ==========================  ==============================================

``standardize_all=True`` additionally standardises every output column
(indicators and flags included) with training statistics. It does not change
what a linear model can represent, but it keeps gradient descent stable when
many 0/1 columns are added (their non-zero means otherwise inflate the largest
eigenvalue of tx.T @ tx / N, and gamma = 0.05 would diverge).

With every option at its default the output is exactly that of v1.

Fit/transform rule: every decision or statistic learned from data belongs in
``params``. ``transform`` computes no fitted statistics of its own. The
codebook rules are not learned (they are fixed facts about the questionnaire)
and are applied identically to every split.
"""

import csv
import itertools
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# v1 settings
# ---------------------------------------------------------------------------
# Candidate codes that can mean "don't know" / "refused" in coded columns.
SENTINEL_CANDIDATES = (7, 9, 77, 99, 777, 999)
# A candidate must occur in less than this share of all training rows.
SENTINEL_MAX_FREQUENCY = 0.02
# Columns missing more than this fraction are dropped rather than imputed.
MAX_MISSING_FRACTION = 0.95
# A kept column with at most this many distinct integer values counts as
# categorical and is one-hot encoded when ``one_hot=True``.
MAX_ONE_HOT_LEVELS = 10

# ---------------------------------------------------------------------------
# v2 settings: codebook rules that the registry's code lists cannot express
# ---------------------------------------------------------------------------
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "docs" / "feature_registry.csv"

# A code that stands for another point of the same scale.
# CHECKUP1: 8 = "never had a checkup" comes after 7 = "don't know"; it belongs
# after 4 = "5 or more years ago".
VALUE_MAPS = {"CHECKUP1": {8.0: 5.0}}

# CDC columns stored x100 in our data (implied decimals not applied).
SCALES = {"DROCDY3_": 0.01, "_DRNKWEK": 0.01}


# ---------------------------------------------------------------------------
# UNITS: raw questions that store several units in one number.
#
# Each rule is a list of (low, high, function) pieces, read from the codebook
# ranges and applied to the RAW code. A code outside every piece (777 = don't
# know, 999 = refused, ...) becomes NaN.
# Example: ALCDAY5 = 104 means 4 days per WEEK -> 4 * 30 / 7 = 17.1 days per
# 30 days, while ALCDAY5 = 204 means 4 days in the past 30 -> 4.
# Without this, a linear model reads 201 (once a month) as more than 107
# (every day), and WEIGHT2 = 9080 (80 kg) as 9,080 pounds.
# ---------------------------------------------------------------------------
def _zero(v):
    return 0.0 * v


def _hours_minutes(v):
    # 130 = 1 h 30 -> 90 minutes
    return (v // 100) * 60 + v % 100


def _month_year(v):
    # 102014 = October 2014 -> 2014.75; an unknown month (77) -> mid-year
    month, year = v // 10000, v % 10000
    out = year + (np.clip(month, 1, 12) - 1) / 12
    out = np.where(month == 77, year + 0.5, out)
    valid = (
        (year >= 1985)
        & (year <= 2016)
        & (((month >= 1) & (month <= 12)) | (month == 77))
    )
    return np.where(valid, out, np.nan)


_PER_DAY = (  # 1xx per day, 2xx per week, 300 less than monthly, 3xx per month, 4xx per year
    (101, 199, lambda v: v - 100),
    (201, 299, lambda v: (v - 200) / 7),
    (300, 300, lambda v: 0.0 * v + 0.5 / 30),
    (301, 399, lambda v: (v - 300) / 30),
    (401, 499, lambda v: (v - 400) / 365),
)
_PER_WEEK = (  # 1xx per week, 2xx per month
    (101, 199, lambda v: v - 100),
    (201, 299, lambda v: (v - 200) * 7 / 30),
)
UNIT_RULES = {
    # pounds; 9000 + kilograms -> pounds
    "WEIGHT2": ((50, 999, lambda v: v), (9000, 9998, lambda v: (v - 9000) * 2.2046)),
    # feet/inches (510 = 5 ft 10) -> inches; 9000 + centimetres -> inches
    "HEIGHT3": (
        (200, 711, lambda v: (v // 100) * 12 + v % 100),
        (9000, 9998, lambda v: (v - 9000) / 2.54),
    ),
    # drinking days per 30 days; 888 = no drinks
    "ALCDAY5": (
        (101, 107, lambda v: (v - 100) * 30 / 7),
        (201, 230, lambda v: v - 200),
        (888, 888, _zero),
    ),
    # times per day; 555 = never (food) / 888 = never (diabetes checks)
    **{
        name: _PER_DAY + ((555, 555, _zero),)
        for name in ("FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1")
    },
    "BLDSUGAR": _PER_DAY + ((888, 888, _zero),),
    "FEETCHK2": _PER_DAY + ((888, 888, _zero),),  # 555 = "no feet" -> NaN
    # times per week; 888 = never (strength training)
    "EXEROFT1": _PER_WEEK,
    "EXEROFT2": _PER_WEEK,
    "STRENGTH": _PER_WEEK + ((888, 888, _zero),),
    # minutes
    "EXERHMM1": ((1, 759, _hours_minutes), (800, 959, _hours_minutes)),
    "EXERHMM2": ((1, 759, _hours_minutes), (800, 959, _hours_minutes)),
    # years: 1xx days, 2xx weeks, 3xx months, 4xx years
    "LONGWTCH": (
        (101, 199, lambda v: (v - 100) / 365),
        (201, 299, lambda v: (v - 200) / 52),
        (301, 399, lambda v: (v - 300) / 12),
        (401, 499, lambda v: v - 400),
    ),
    # decimal year
    "FLSHTMY2": ((10000, 999999, _month_year),),
    "HIVTSTD3": ((10000, 999999, _month_year),),
}

# Skip logic: (column, condition on the RAW row, value) - a blank in `column`
# means "not asked because of an earlier answer", so the answer is implied.
# Conditions read raw codes, before any recoding of the conditioning column.
SKIP_RULES = (
    # no high blood pressure -> not taking blood-pressure medication (2 = no)
    ("BPMEDS", ("BPHIGH4", (2.0, 3.0, 4.0)), 2.0),
    # cholesterol never checked -> new level 3 "never checked"
    ("TOLDHI2", ("BLOODCHO", (2.0,)), 3.0),
    # no bad physical and no bad mental days -> 0 days kept from activities
    ("POORHLTH", ("PHYSHLTH", (88.0,)), 0.0),
    # no drinks in the past 30 days -> no binge, no maximum
    ("DRNK3GE5", ("ALCDAY5", (888.0,)), 0.0),
    ("MAXDRNKS", ("ALCDAY5", (888.0,)), 0.0),
    # no exercise -> 0 minutes of activity
    ("PA1MIN_", ("EXERANY2", (2.0,)), 0.0),
    ("PA1VIGM_", ("EXERANY2", (2.0,)), 0.0),
    # no arthritis -> no joint pain question -> 0
    ("JOINPAIN", ("HAVARTH3", (2.0,)), 0.0),
)
# POORHLTH is only skipped when BOTH PHYSHLTH and MENTHLTH are 88.
SKIP_ALSO = {"POORHLTH": ("MENTHLTH", (88.0,))}

# Module questions that are blank for almost everyone: the blank itself is the
# signal. HAREHAB1 is only asked to people who had a heart attack (all 773
# training rows that answered it have MICHD = 1). name -> also add "answered yes".
MODULE_FLAGS = (("HAREHAB1", False), ("CVDASPRN", True), ("RDUCHART", True))

# Step 4: heavy-tailed counts get clip (training percentile) + log1p.
HEAVY_TAILED = (
    "PA1MIN_",
    "PA1VIGM_",
    "_DRNKWEK",
    "DROCDY3_",
    "STRFREQ_",
    "MAXDRNKS",
    "DRNK3GE5",
    "CHILDREN",
    "FTJUDA1_",
    "FRUTDA1_",
    "BEANDAY_",
    "GRENDAY_",
    "ORNGDAY_",
    "VEGEDA1_",
)
CLIP_PERCENTILE = 99.5
# Step 4: curved risk -> add the square of the standardised column.
SQUARED = ("_AGE80", "_BMI5")
# Step 5: categorical with more than MAX_ONE_HOT_LEVELS levels.
STATE = "_STATE"

REGISTRY_OPTIONS = (
    "codes",
    "flags",
    "registry_one_hot",
    "select",
    "nonlinear",
    "extras",
    "engineered",
    "aspirin_interactions",
)

# ---------------------------------------------------------------------------
# STEP 7: ENGINEERED COLUMNS, for what a linear model cannot build itself.
#
# The score is a weighted sum of the columns, so a sum of existing columns
# (a plain count of risk factors) adds nothing. A product (an interaction,
# "old AND diabetic") or a curve of a sum (count^2) it cannot build. Each
# group was measured on its own in experiments/261004_aa_preprocessing_v2.ipynb
# (step 7). All are fixed formulas of the codebook-cleaned answers except
# "ordinal_one_hot", whose levels are learned on the training rows.
#
#   sex_age          male*age, male*age^2: the age slope is steeper for men
#   sex_factors      male x each of SEX_FACTORS
#   age_factors      age x each of the 12 RISK_FACTORS
#   counts           CARDIO, CHRONIC and DISABILITY counts, each with its square
#   genhlth          general health (GENHLTH - 3) x age, x male, x each risk
#                    factor, and its square: risk curves up from good to poor
#   ordinal_one_hot  one 0/1 column per level of each numeric column with
#                    3..ORDINAL_MAX_LEVELS levels (CHECKUP1 is not monotone)
#   factor_pairs     every product of two PAIR_FACTORS (high BP x high chol)
#   sex_one_hot      male x every registry one-hot indicator
#
# ENGINEERED_DEFAULT (what engineered=True adds) leaves out ordinal_one_hot and
# sex_one_hot: 572 of the 703 engineered columns, and with logistic regression
# on all rows they added nothing: F1 0.4294 without them (run
# 261011_aa_v2-engineered-lean-3), 0.4293 with them (exploratory run
# 261011_aa_v2-engineered-2), for 1.5x the time.
#
# age = (_AGE80 - 50) / 10, a fixed formula, so a product does not just copy
# its factor. A factor that is NaN (don't know, refused, not asked) counts as
# "no". Use with standardize_all=True: the new columns are then standardised
# with training statistics like every other column.
# ---------------------------------------------------------------------------
ENGINEERED_GROUPS = (
    "sex_age",
    "sex_factors",
    "age_factors",
    "counts",
    "genhlth",
    "ordinal_one_hot",
    "factor_pairs",
    "sex_one_hot",
)
ENGINEERED_DEFAULT = tuple(
    g for g in ENGINEERED_GROUPS if g not in ("ordinal_one_hot", "sex_one_hot")
)
AGE_CENTRE, AGE_SCALE = 50.0, 10.0
OBESE_BMI = 30.0
ORDINAL_MAX_LEVELS = 15
# name, column, codes that mean "yes" (None for obese: BMI >= OBESE_BMI)
RISK_FACTORS = (
    ("high_bp", "BPHIGH4", (1.0,)),
    ("high_chol", "TOLDHI2", (1.0,)),
    ("diabetes", "DIABETE3", (1.0,)),
    ("ever_smoked", "_SMOKER3", (1.0, 2.0, 3.0)),
    ("obese", "_BMI5", None),
    ("inactive", "_TOTINDA", (2.0,)),
    ("stroke", "CVDSTRK3", (1.0,)),
    ("kidney", "CHCKIDNY", (1.0,)),
    ("copd", "CHCCOPD1", (1.0,)),
    ("arthritis", "HAVARTH3", (1.0,)),
    ("depression", "ADDEPEV2", (1.0,)),
    ("diff_walk", "DIFFWALK", (1.0,)),
)
# factor_pairs also uses these two
PAIR_EXTRA_FACTORS = (
    ("poor_health", "GENHLTH", (4.0, 5.0)),
    ("bp_meds", "BPMEDS", (1.0,)),
)
SEX_FACTORS = ("high_bp", "diabetes", "high_chol", "ever_smoked", "diff_walk")
CARDIO = ("high_bp", "high_chol", "diabetes", "ever_smoked", "obese", "inactive")
CHRONIC = ("stroke", "kidney", "copd", "diabetes", "arthritis", "depression")
DISABILITY = (
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
    "USEEQUIP",
    "QLACTLM2",
    "DECIDE",
    "BLIND",
)

# ---------------------------------------------------------------------------
# STEP 8 (experimental): ASPIRIN-MODULE INTERACTIONS.
#
# CVDASPRN ("take aspirin daily?") belongs to an optional cardiovascular module
# that only some states ask (about 4% of rows). Near the decision cut-off
# (out-of-fold risk 12-45%, every 3rd row) the step-7 model over-predicted the
# people who answered it: 14% positive against 21% predicted (about 300 rows).
# aspirin_interactions=True lets the main risk factors weigh differently
# inside that group: answered x age, x male, x general health (GENHLTH - 3)
# and x each of ASPIRIN_FACTORS; all 0 for people not asked.
#
# Measured on top of the step-7 recipe with the tuned logistic model: +0.0019
# on every 3rd row (the rows the misfit was found on), +0.0007 on all rows.
# Caveat: daily aspirin is often a consequence of heart disease, so like
# HAREHAB1 (step 2) the answer partly reveals the label.
# ---------------------------------------------------------------------------
ASPIRIN_QUESTION = "CVDASPRN"
ASPIRIN_FACTORS = ("high_bp", "high_chol", "diabetes", "bp_meds")


# ---------------------------------------------------------------------------
# REGISTRY: the per-column decisions of docs/feature_registry.csv.
# ---------------------------------------------------------------------------
def load_registry(path=REGISTRY_PATH):
    """Read the feature registry into one dict per column, in column order.

    Each dict has ``name``, ``index``, ``decision`` and the parsed code lists
    ``nan_codes`` / ``zero_codes`` (floats). Only the standard library is used.
    """
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "name": row["column"],
                    "index": int(row["index"]),
                    "decision": row["decision"],
                    "nan_codes": [float(c) for c in row.get("nan_codes", "").split()],
                    "zero_codes": [float(c) for c in row.get("zero_codes", "").split()],
                }
            )
    rows.sort(key=lambda r: r["index"])
    return rows


def _compile_rules(registry, n_cols, extras):
    """Turn the registry into index-based rules stored in ``params``.

    ``transform`` must not read files or learn anything, so everything it
    needs from the registry is copied into this plain dict once, at fit time.
    """
    if len(registry) != n_cols:
        raise ValueError(
            f"the registry describes {len(registry)} columns but x has {n_cols}"
        )
    index = {r["name"]: r["index"] for r in registry}
    return {
        "index": index,
        "nan": {r["index"]: r["nan_codes"] for r in registry if r["nan_codes"]},
        "zero": {r["index"]: r["zero_codes"] for r in registry if r["zero_codes"]},
        "decision": {r["index"]: r["decision"] for r in registry},
        "extras": bool(extras),
    }


# ---------------------------------------------------------------------------
# APPLY CODEBOOK, the fixed rules of step 1 (and the derived columns of step 5).
#
# Example for one row, PHYSHLTH = 88, BPHIGH4 = 3, BPMEDS = blank, WEIGHT2 = 9080:
#   PHYSHLTH 88 means "none"                     -> 0
#   BPMEDS is blank because BPHIGH4 = 3 ("no")   -> 2 ("no medication")
#   WEIGHT2 9080 means 80 kg                     -> 176.4 (pounds)
# Nothing here depends on other rows, so it is safe on any split.
# ---------------------------------------------------------------------------
def apply_codebook(x, rules):
    """Return a copy of ``x`` with the codebook rules applied (learns nothing)."""
    raw = np.asarray(x, dtype=float)
    out = raw.copy()
    idx = rules["index"]

    for j, codes in rules["nan"].items():
        out[np.isin(raw[:, j], codes), j] = np.nan
    for j, codes in rules["zero"].items():
        out[np.isin(raw[:, j], codes), j] = 0.0
    for name, mapping in VALUE_MAPS.items():
        j = idx[name]
        for old, new in mapping.items():
            out[raw[:, j] == old, j] = new
    for name, factor in SCALES.items():
        out[:, idx[name]] *= factor
    for name, pieces in UNIT_RULES.items():
        j = idx[name]
        converted = np.full(len(raw), np.nan)
        for low, high, convert in pieces:
            inside = (raw[:, j] >= low) & (raw[:, j] <= high)
            converted[inside] = convert(raw[inside, j])
        out[:, j] = converted

    for name, (cond_name, cond_values), value in SKIP_RULES:
        j = idx[name]
        skipped = np.isnan(raw[:, j]) & np.isin(raw[:, idx[cond_name]], cond_values)
        if name in SKIP_ALSO:
            also_name, also_values = SKIP_ALSO[name]
            skipped &= np.isin(raw[:, idx[also_name]], also_values)
        out[skipped, j] = value

    if rules["extras"]:
        # Years with diabetes, stored in the DIABAGE2 column: age now minus
        # age at diagnosis for diabetics, 0 for everyone told they are not.
        j = idx["DIABAGE2"]
        diabetic = raw[:, idx["DIABETE3"]] == 1
        not_diabetic = np.isin(raw[:, idx["DIABETE3"]], (2.0, 3.0, 4.0))
        years = np.clip(raw[:, idx["_AGE80"]] - out[:, j], 0, None)
        out[:, j] = np.where(diabetic, years, np.where(not_diabetic, 0.0, np.nan))
        # One "adults in household" column: NUMADULT (landline survey) or
        # HHADULT (cell-phone survey), capped at 6.
        j, k = idx["NUMADULT"], idx["HHADULT"]
        merged = np.where(np.isnan(out[:, j]), out[:, k], out[:, j])
        out[:, j] = np.minimum(merged, 6.0)
    return out


def _module_flags(x, flag_spec):
    """0/1 flags built from the raw row: "was asked" and "answered yes"."""
    x = np.asarray(x, dtype=float)
    cols = []
    for j, kind in flag_spec:
        if kind == "asked":
            cols.append(~np.isnan(x[:, j]))
        else:
            cols.append(x[:, j] == 1.0)
    return np.column_stack(cols + [np.empty((len(x), 0))]).astype(float)


# ---------------------------------------------------------------------------
# SENTINEL HEURISTIC (v1), used when codes=False.
#
# Three columns from the full training set show where each check decides
# (measured 3 Oct 2026; see docs/feature_registry.csv):
# - GENHLTH {1..5, 7, 9}: 7 and 9 are rare (0.2 %, 0.1 %), the rest is
#   1..5 with a gap before 7 -> both detected, correctly.
# - EMPLOY1 {1..9}: 7 = "retired" is a real answer but is removed as a
#   candidate too, so the rest {1..6, 8} has a hole -> not contiguous,
#   and 9 = "refused" is missed.
# - PHYSHLTH {1..30, 77, 88, 99}: 88 = "none" is no candidate at all, and
#   removing 7 and 9 (real day counts) leaves holes -> nothing detected.
# The registry (codes=True) replaces this guess with the codebook.
# ---------------------------------------------------------------------------
def _detect_sentinels(working):
    n_rows, n_cols = working.shape
    sentinels = {}
    candidates = np.asarray(SENTINEL_CANDIDATES)

    for column_index in range(n_cols):
        column = working[:, column_index]
        observed = column[~np.isnan(column)]
        if observed.size == 0:
            continue

        detected = []
        for candidate in SENTINEL_CANDIDATES:
            candidate_count = np.count_nonzero(observed == candidate)
            if candidate_count == 0:
                continue
            # Check 1, rare: a code given by 2 % or more of everyone is taken
            # to be a real answer (EMPLOY1 7 = retired, 29.9 %, stops here).
            if candidate_count / n_rows >= SENTINEL_MAX_FREQUENCY:
                continue
            genuine = np.unique(observed[~np.isin(observed, candidates)])
            if genuine.size == 0:
                continue
            # Check 2, coded answers: integer-valued and beginning at 0 or 1.
            if not np.allclose(genuine, np.round(genuine)):
                continue
            genuine = np.round(genuine).astype(int)
            if genuine[0] not in (0, 1):
                continue
            # Check 3, contiguous, and check 4, a gap before the candidate.
            contiguous = np.array_equal(genuine, np.arange(genuine[0], genuine[-1] + 1))
            if contiguous and genuine[-1] <= candidate - 2:
                detected.append(candidate)

        if detected:
            sentinels[column_index] = tuple(detected)
            working[np.isin(working[:, column_index], detected), column_index] = np.nan
    return sentinels


# ---------------------------------------------------------------------------
# FIT PREPROCESSOR, learning how to clean and represent the training features.
#
# Order of the steps (each one only when its option is on):
#  a. codebook rules (codes) or the v1 sentinel heuristic
#  b. clip + log1p of heavy-tailed counts (nonlinear): clip values learned here
#  c. column choice: registry non-drop columns (select) or the v1 rule
#     (drop all-NaN, > 95 % missing and constant columns)
#  d. median, then mean/std after imputation           (as v1)
#  e. one-hot positions and their training levels      (registry or v1 rule)
#  f. missing-indicator positions                      (as v1)
#  g. statistics of powers (v1 poly_degree) and of squares (nonlinear)
#  h. module flags (flags)
#  i. engineered groups and the levels of ordinal_one_hot (engineered),
#     aspirin-module interactions (aspirin_interactions)
#  j. mean/std of every output column (standardize_all)
# ---------------------------------------------------------------------------
def fit_preprocessor(
    x_train,
    one_hot=False,
    poly_degree=1,
    missing_indicators=False,
    codes=False,
    flags=False,
    registry_one_hot=False,
    select=False,
    nonlinear=False,
    extras=False,
    engineered=(),
    aspirin_interactions=False,
    standardize_all=False,
    registry_path=None,
):
    """Learn preprocessing parameters from the training matrix.

    Args:
        x_train (np.ndarray): raw training features, shape (N, D). Never ``y``.
        one_hot, poly_degree, missing_indicators: v1 options (see module doc).
        codes, flags, registry_one_hot, nonlinear, extras, select: v2 steps
            1-6 (see module doc); ``registry_one_hot``, ``select``,
            ``nonlinear`` and ``extras`` require ``codes=True``.
        engineered (tuple of str or True): step 7, the ENGINEERED_GROUPS to
            add (True = ENGINEERED_DEFAULT, the six recommended groups);
            requires ``codes=True``.
        aspirin_interactions (bool): step 8 (experimental), the aspirin-module
            answer x age, sex, general health and ASPIRIN_FACTORS; requires
            ``codes=True``.
        standardize_all (bool): standardise every output column.
        registry_path (str or Path): registry CSV; defaults to
            ``docs/feature_registry.csv``.

    Returns:
        dict: ``params`` for :func:`transform`. Beyond the v1 keys
        (``keep_cols``, ``sentinels``, ``median``, ``mean``, ``std``,
        ``numeric_pos``, ``one_hot_pos``, ``levels``, ``missing_level``,
        ``poly_degree``, ``poly_mean``, ``poly_std``,
        ``missing_indicator_pos``) it may hold ``rules`` (compiled codebook
        rules), ``log_clip`` ({column: clip value}), ``square_pos`` with
        ``square_mean`` / ``square_std``, ``flag_spec``, ``engineered`` with
        ``ordinal_levels``, ``aspirin_interactions`` and ``out_mean`` /
        ``out_std``.
    """
    x_train = np.asarray(x_train, dtype=float)
    if x_train.ndim != 2:
        raise ValueError("x_train must be a two-dimensional array")
    if x_train.shape[0] == 0:
        raise ValueError("x_train must contain at least one row")
    if int(poly_degree) != poly_degree or poly_degree < 1:
        raise ValueError("poly_degree must be an integer >= 1")
    groups = ENGINEERED_DEFAULT if engineered is True else tuple(engineered or ())
    unknown = set(groups) - set(ENGINEERED_GROUPS)
    if unknown:
        raise ValueError(f"unknown engineered groups: {sorted(unknown)}")
    needs_codes = registry_one_hot or select or nonlinear or extras or groups
    if (needs_codes or aspirin_interactions) and not codes:
        raise ValueError(
            "registry_one_hot, select, nonlinear, extras, engineered and "
            "aspirin_interactions need codes=True"
        )

    n_rows, n_cols = x_train.shape
    use_registry = codes or flags
    rules = None
    if use_registry:
        registry = load_registry(registry_path or REGISTRY_PATH)
        rules = _compile_rules(registry, n_cols, extras and codes)
        index = rules["index"]

    # a. codes ---------------------------------------------------------------
    if codes:
        working = apply_codebook(x_train, rules)
        sentinels = {}
    else:
        working = x_train.copy()
        sentinels = _detect_sentinels(working)

    # b. heavy tails -----------------------------------------------------------
    log_clip = {}
    if nonlinear:
        for name in HEAVY_TAILED:
            j = index[name]
            column = working[:, j]
            if np.all(np.isnan(column)):
                continue
            log_clip[j] = float(np.nanpercentile(column, CLIP_PERCENTILE))
        _apply_log_clip(working, log_clip)

    # c. column choice ---------------------------------------------------------
    missing_fraction = np.isnan(working).mean(axis=0)
    if select:
        excluded = {index["HAREHAB1"]}  # its information is the "asked" flag
        if extras:
            excluded.add(index["HHADULT"])  # merged into NUMADULT
        else:
            excluded.add(index[STATE])  # a state code is meaningless as a number
        candidates = [
            j
            for j in range(n_cols)
            if rules["decision"][j] != "drop" and j not in excluded
        ]
    else:
        candidates = np.where(
            (missing_fraction < 1.0) & (missing_fraction <= MAX_MISSING_FRACTION)
        )[0]
    keep_cols = []
    for j in candidates:
        if missing_fraction[j] == 1.0:
            continue
        column_std = np.nanstd(working[:, j])
        if np.isfinite(column_std) and column_std > 0:
            keep_cols.append(j)
    keep_cols = np.asarray(keep_cols, dtype=int)
    if keep_cols.size == 0:
        raise ValueError("No usable columns remain after preprocessing")

    # d. imputation and standardisation statistics ------------------------------
    kept_data = working[:, keep_cols]
    median = np.nanmedian(kept_data, axis=0)
    imputed = np.where(np.isnan(kept_data), median, kept_data)
    mean = imputed.mean(axis=0)
    std = imputed.std(axis=0)
    std[std == 0] = 1.0

    # e. one-hot ----------------------------------------------------------------
    if registry_one_hot:
        one_hot_names = {
            j
            for j, decision in rules["decision"].items()
            if decision == "one-hot" and (extras or j != index[STATE])
        }
        positions = [p for p, j in enumerate(keep_cols) if j in one_hot_names]
    elif one_hot:
        positions = []
        for p in range(keep_cols.size):
            values = np.unique(kept_data[~np.isnan(kept_data[:, p]), p])
            if values.size <= MAX_ONE_HOT_LEVELS and np.allclose(
                values, np.round(values)
            ):
                positions.append(p)
    else:
        positions = []
    one_hot_pos = np.asarray(positions, dtype=int)
    levels, missing_level = [], []
    for p in one_hot_pos:
        column = kept_data[:, p]
        levels.append(np.unique(column[~np.isnan(column)]))
        missing_level.append(bool(np.isnan(column).any()))
    numeric_pos = np.setdiff1d(np.arange(keep_cols.size), one_hot_pos)

    # f. missing indicators -------------------------------------------------------
    if missing_indicators:
        missing_indicator_pos = np.setdiff1d(
            np.flatnonzero(np.isnan(kept_data).any(axis=0)), one_hot_pos
        )
    else:
        missing_indicator_pos = np.array([], dtype=int)

    # g. powers and squares -------------------------------------------------------
    standardised = (imputed[:, numeric_pos] - mean[numeric_pos]) / std[numeric_pos]
    powers = _powers(standardised, int(poly_degree))
    poly_mean = powers.mean(axis=0)
    poly_std = powers.std(axis=0)
    poly_std[poly_std == 0] = 1.0

    params = {
        "keep_cols": keep_cols,
        "sentinels": sentinels,
        "median": median,
        "mean": mean,
        "std": std,
        "numeric_pos": numeric_pos,
        "one_hot_pos": one_hot_pos,
        "levels": levels,
        "missing_level": missing_level,
        "poly_degree": int(poly_degree),
        "poly_mean": poly_mean,
        "poly_std": poly_std,
        "missing_indicator_pos": missing_indicator_pos,
    }
    if codes:
        params["rules"] = rules
    if log_clip:
        params["log_clip"] = log_clip
    if nonlinear:
        # positions within the numeric block of the columns to square
        wanted = {index[name] for name in SQUARED}
        square_pos = np.array(
            [q for q, p in enumerate(numeric_pos) if keep_cols[p] in wanted], dtype=int
        )
        squares = standardised[:, square_pos] ** 2
        params["square_pos"] = square_pos
        params["square_mean"] = squares.mean(axis=0)
        params["square_std"] = np.where(
            squares.std(axis=0) == 0, 1.0, squares.std(axis=0)
        )

    # h. module flags -------------------------------------------------------------
    if flags:
        spec = []
        for name, with_yes in MODULE_FLAGS:
            spec.append((index[name], "asked"))
            if with_yes:
                spec.append((index[name], "yes"))
        params["flag_spec"] = spec

    # i. engineered columns ------------------------------------------------------
    if groups:
        # kept in ENGINEERED_GROUPS order, so the layout never depends on how
        # the option was written
        params["engineered"] = tuple(g for g in ENGINEERED_GROUPS if g in groups)
        if "ordinal_one_hot" in groups:
            ordinal = []
            for p in numeric_pos:
                column = kept_data[:, p]
                values = np.unique(column[~np.isnan(column)])
                if 3 <= values.size <= ORDINAL_MAX_LEVELS:
                    ordinal.append((int(keep_cols[p]), values))
            params["ordinal_levels"] = ordinal
    if aspirin_interactions:
        params["aspirin_interactions"] = True

    # j. standardise every output column ---------------------------------------------
    if standardize_all:
        features = _features(x_train, params)
        out_mean = features.mean(axis=0)
        out_std = features.std(axis=0)
        out_std[out_std == 0] = 1.0
        params["out_mean"] = out_mean
        params["out_std"] = out_std
    return params


def _apply_log_clip(working, log_clip):
    """In place: clip each listed column at its training value, then log1p."""
    for j, upper in log_clip.items():
        working[:, j] = np.log1p(np.clip(working[:, j], 0.0, upper))


# ---------------------------------------------------------------------------
# POWERS, creating polynomial features from the numeric feature matrix z.
#
# For example, a row z = [2, 3] with degree = 3 produces:
# [2**2, 3**2, 2**3, 3**3] = [4, 9, 8, 27].
# All squares come first, followed by all cubes. Degree 1 produces no columns.
# ---------------------------------------------------------------------------
def _powers(z, degree):
    """Return ``[z**2, ..., z**degree]`` side by side; no columns for degree 1."""
    return np.hstack([z**power for power in range(2, degree + 1)] + [z[:, :0]])


# ---------------------------------------------------------------------------
# ONE-HOT ENCODING, one 0/1 column per category learned in training.
#
# Training categories [1, 2, 3], value 2 -> [0, 1, 0]. A NaN activates the
# extra "missing / unknown" column when training had NaN in that column.
# An unseen category gives zeros, so all splits keep the same layout.
# ---------------------------------------------------------------------------
def _one_hot(selected, params):
    """Return the 0/1 indicator block for the one-hot columns of ``selected``."""
    blocks = []
    for position, values, has_missing in zip(
        params["one_hot_pos"], params["levels"], params["missing_level"]
    ):
        column = selected[:, position]
        blocks.append((column[:, None] == values[None, :]).astype(float))
        if has_missing:
            blocks.append(np.isnan(column)[:, None].astype(float))
    return np.hstack(blocks + [np.empty((len(selected), 0))])


def _features(x, params):
    """Every output column except the intercept, before ``standardize_all``.

    Column order: numeric, powers, squares, one-hot indicators, missing flags,
    module flags, engineered columns, aspirin interactions. Shared by
    :func:`fit_preprocessor` (for ``standardize_all``)
    and :func:`transform`, so both always build the same layout.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional array")

    if "rules" in params:
        working = apply_codebook(x, params["rules"])
    else:
        working = x.copy()
        for column_index, values in params["sentinels"].items():
            if column_index < 0 or column_index >= working.shape[1]:
                raise ValueError("x does not have the columns expected by params")
            working[np.isin(working[:, column_index], values), column_index] = np.nan
    if "log_clip" in params:
        _apply_log_clip(working, params["log_clip"])

    keep_cols = np.asarray(params["keep_cols"], dtype=int)
    if keep_cols.size and (keep_cols.min() < 0 or keep_cols.max() >= working.shape[1]):
        raise ValueError("x does not have the columns expected by params")
    selected = working[:, keep_cols]

    median = np.asarray(params["median"], dtype=float)
    mean = np.asarray(params["mean"], dtype=float)
    std = np.asarray(params["std"], dtype=float)
    for name, values in (("median", median), ("mean", mean), ("std", std)):
        if values.shape != (keep_cols.size,):
            raise ValueError(f'params["{name}"] must have shape {(keep_cols.size,)}')
    if np.any(std == 0) or not np.all(np.isfinite(std)):
        raise ValueError('params["std"] must contain finite, non-zero values')

    # Parameters created before the v1 options omit these keys; the defaults
    # reproduce the original numeric pipeline.
    numeric_pos = np.asarray(
        params.get("numeric_pos", np.arange(keep_cols.size)), dtype=int
    )
    poly_degree = params.get("poly_degree", 1)

    imputed = np.where(np.isnan(selected), median, selected)
    standardised = (imputed[:, numeric_pos] - mean[numeric_pos]) / std[numeric_pos]
    powers = _powers(standardised, poly_degree)
    if poly_degree > 1:
        powers = (powers - params["poly_mean"]) / params["poly_std"]

    if "square_pos" in params:
        squares = standardised[:, params["square_pos"]] ** 2
        squares = (squares - params["square_mean"]) / params["square_std"]
    else:
        squares = np.empty((len(x), 0))

    indicators = (
        _one_hot(selected, params) if "one_hot_pos" in params else np.empty((len(x), 0))
    )

    missing_indicator_pos = np.asarray(
        params.get("missing_indicator_pos", []), dtype=int
    )
    if missing_indicator_pos.size and (
        missing_indicator_pos.min() < 0 or missing_indicator_pos.max() >= keep_cols.size
    ):
        raise ValueError('params["missing_indicator_pos"] contains an invalid position')
    # Flags read `selected` before imputation, so they still see the NaN.
    missing_flags = np.isnan(selected[:, missing_indicator_pos]).astype(float)

    module = (
        _module_flags(x, params["flag_spec"])
        if "flag_spec" in params
        else np.empty((len(x), 0))
    )

    engineered = (
        _engineered(working, indicators, params)[0]
        if "engineered" in params
        else np.empty((len(x), 0))
    )
    aspirin = (
        _aspirin_interactions(working, params)[0]
        if params.get("aspirin_interactions")
        else np.empty((len(x), 0))
    )

    return np.hstack(
        [
            standardised,
            powers,
            squares,
            indicators,
            missing_flags,
            module,
            engineered,
            aspirin,
        ]
    )


# ---------------------------------------------------------------------------
# ENGINEERED COLUMNS (step 7), from the codebook-cleaned answers.
#
# Example, a 70-year-old man with high blood pressure and diabetes:
#   age = (70 - 50) / 10 = 2   male x age = 2   age x high_bp = 2
#   high_bp x diabetes = 1     # cardio counts both -> 2, and 2^2 = 4
# ---------------------------------------------------------------------------
def risk_factors(working, index):
    """RISK_FACTORS then PAIR_EXTRA_FACTORS as 0/1 columns (NaN counts as no).

    ``working`` is the full-width matrix after :func:`apply_codebook`,
    ``index`` maps column names to positions (``params["rules"]["index"]``).
    """
    factors = {}
    for name, column, yes in RISK_FACTORS + PAIR_EXTRA_FACTORS:
        values = working[:, index[column]]
        hit = values >= OBESE_BMI if yes is None else np.isin(values, yes)
        factors[name] = hit.astype(float)
    return factors


def _raw_name(params, j):
    """Name of raw column ``j`` (from the registry when the codes are used)."""
    if "rules" in params:
        return next(n for n, k in params["rules"]["index"].items() if k == j)
    return f"x{j}"


def _one_hot_labels(params):
    """'NAME = level' for every one-hot indicator column, in transform's order."""
    labels = []
    for p, values, has_missing in zip(
        params["one_hot_pos"], params["levels"], params["missing_level"]
    ):
        name = _raw_name(params, int(params["keep_cols"][p]))
        labels += [f"{name} = {v:g}" for v in values]
        if has_missing:
            labels.append(f"{name} = unknown")
    return labels


def _engineered(working, indicators, params):
    """Step 7 columns and their names, for the groups in ``params["engineered"]``.

    ``working``: full-width codebook-cleaned rows; ``indicators``: the one-hot
    block of :func:`_one_hot` (used by sex_one_hot).
    """
    index = params["rules"]["index"]
    groups = params["engineered"]
    f = risk_factors(working, index)
    risk = [name for name, _, _ in RISK_FACTORS]
    age = np.nan_to_num(working[:, index["_AGE80"]], nan=AGE_CENTRE)
    age = (age - AGE_CENTRE) / AGE_SCALE
    male = (working[:, index["SEX"]] == 1).astype(float)
    cols, names = [], []

    def add(column, name):
        cols.append(column)
        names.append(name)

    if "sex_age" in groups:
        add(male * age, "male x age")
        add(male * age**2, "male x age^2")
    if "sex_factors" in groups:
        for n in SEX_FACTORS:
            add(male * f[n], f"male x {n}")
    if "age_factors" in groups:
        for n in risk:
            add(age * f[n], f"age x {n}")
    if "counts" in groups:
        disability = sum((working[:, index[c]] == 1).astype(float) for c in DISABILITY)
        for label, count in (
            ("cardio", sum(f[n] for n in CARDIO)),
            ("chronic", sum(f[n] for n in CHRONIC)),
            ("disability", disability),
        ):
            add(count, f"# {label}")
            add(count**2, f"# {label}^2")
    if "genhlth" in groups:
        health = np.nan_to_num(working[:, index["GENHLTH"]] - 3.0)  # 0 = good
        add(health * age, "genhlth x age")
        add(health * male, "genhlth x male")
        add(health**2, "genhlth^2")
        for n in risk:
            add(health * f[n], f"genhlth x {n}")
    if "ordinal_one_hot" in groups:
        for j, values in params["ordinal_levels"]:
            for v in values:
                add(
                    (working[:, j] == v).astype(float),
                    f"{_raw_name(params, j)} = {v:g}",
                )
    if "factor_pairs" in groups:
        for a, b in itertools.combinations(f, 2):
            add(f[a] * f[b], f"{a} x {b}")
    if "sex_one_hot" in groups:
        for k, label in enumerate(_one_hot_labels(params)):
            add(male * indicators[:, k], f"male x {label}")
    return np.column_stack(cols + [np.empty((len(working), 0))]), names


def _aspirin_interactions(working, params):
    """Step 8 columns and their names: answered CVDASPRN x age, sex, health, factors.

    "Answered" means a yes or no after the codebook codes (don't know and
    refused are NaN), so it is 0 for everyone outside the aspirin module.
    """
    index = params["rules"]["index"]
    answered = (~np.isnan(working[:, index[ASPIRIN_QUESTION]])).astype(float)
    f = risk_factors(working, index)
    age = np.nan_to_num(working[:, index["_AGE80"]], nan=AGE_CENTRE)
    age = (age - AGE_CENTRE) / AGE_SCALE
    male = (working[:, index["SEX"]] == 1).astype(float)
    health = np.nan_to_num(working[:, index["GENHLTH"]] - 3.0)  # 0 = good
    cols = [answered * age, answered * male, answered * health]
    cols += [answered * f[n] for n in ASPIRIN_FACTORS]
    names = [f"aspirin answered x {n}" for n in ("age", "male", "genhlth")]
    names += [f"aspirin answered x {n}" for n in ASPIRIN_FACTORS]
    return np.column_stack(cols), names


def column_layout(params):
    """Describe every column of ``transform(x, params)``, in order.

    Returns a list of ``(kind, source, name)``: ``kind`` is one of intercept,
    numeric, power, square, one-hot, missing, flag, engineered, aspirin;
    ``source`` is the raw column the output is built from (-1 for the
    intercept and for engineered and aspirin columns, which combine several).
    """
    keep_cols = np.asarray(params["keep_cols"], dtype=int)
    numeric_pos = np.asarray(
        params.get("numeric_pos", np.arange(keep_cols.size)), dtype=int
    )
    logged = params.get("log_clip", {})

    def base(j):
        name = _raw_name(params, j)
        return f"log({name})" if j in logged else name

    layout = [("intercept", -1, "intercept")]
    layout += [("numeric", int(keep_cols[p]), base(keep_cols[p])) for p in numeric_pos]
    for degree in range(2, params.get("poly_degree", 1) + 1):
        layout += [
            ("power", int(keep_cols[p]), f"{base(keep_cols[p])}^{degree}")
            for p in numeric_pos
        ]
    for q in params.get("square_pos", []):
        j = int(keep_cols[numeric_pos[q]])
        layout.append(("square", j, f"{base(j)}^2"))
    if "one_hot_pos" in params:
        sources = []
        for p, values, has_missing in zip(
            params["one_hot_pos"], params["levels"], params["missing_level"]
        ):
            sources += [int(keep_cols[p])] * (len(values) + int(has_missing))
        layout += [
            ("one-hot", j, label) for j, label in zip(sources, _one_hot_labels(params))
        ]
    for p in params.get("missing_indicator_pos", []):
        j = int(keep_cols[p])
        layout.append(("missing", j, f"{_raw_name(params, j)} missing"))
    for j, kind in params.get("flag_spec", []):
        layout.append(("flag", j, f"{_raw_name(params, j)} {kind}"))
    if "engineered" in params:
        n_indicators = sum(
            len(v) + int(m) for v, m in zip(params["levels"], params["missing_level"])
        )
        empty = np.full((1, len(params["rules"]["index"])), np.nan)
        _, names = _engineered(empty, np.zeros((1, n_indicators)), params)
        layout += [("engineered", -1, name) for name in names]
    if params.get("aspirin_interactions"):
        empty = np.full((1, len(params["rules"]["index"])), np.nan)
        _, names = _aspirin_interactions(empty, params)
        layout += [("aspirin", -1, name) for name in names]
    return layout


# ---------------------------------------------------------------------------
# TRANSFORM, applying the learned preprocessing to any data split.
#
# For a numeric value z = (filled value - training mean) / training std, with
# missing values first replaced by the stored training median. A leading 1
# lets a linear model learn an intercept: prediction = w_0 + w_1 * f_1 + ...
# ---------------------------------------------------------------------------
def transform(x, params):
    """Apply learned preprocessing and return the design matrix ``tx``.

    Column order: intercept, numeric, powers, squares, one-hot indicators,
    missing flags, module flags, engineered columns (:func:`column_layout`
    names them). Output is NaN-free; ``transform`` never reads
    ``y`` and never computes a statistic of its own.

    Invariants worth asserting in tests:

    - no NaN and no inf in the output
    - ``transform(x_train, params)`` and ``transform(x_test, params)`` have the
      same number of columns
    - calling it twice gives identical results (no hidden state)
    """
    required_keys = {"keep_cols", "sentinels", "median", "mean", "std"}
    missing_keys = required_keys.difference(params)
    if missing_keys:
        raise ValueError(f"params is missing keys: {sorted(missing_keys)}")

    features = _features(x, params)
    if "out_mean" in params:
        features -= params["out_mean"]
        features /= params["out_std"]
    tx = np.hstack([np.ones((len(features), 1)), features])

    if not np.all(np.isfinite(tx)):
        raise ValueError("transform produced non-finite values")
    return tx
