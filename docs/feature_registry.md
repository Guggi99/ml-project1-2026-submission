# Feature registry — what each column really means (#6)

[`feature_registry.csv`](feature_registry.csv) has one row per column of
`x_train` (321 rows). It is how we turn the CDC codebook into preprocessing
rules: we write down what the codes mean, then the code applies the table.

**Why it matters:** today `src/preprocessing.py` guesses special codes. It
misses many of them — `PHYSHLTH` is "88" for 62 % of people, which means
**zero** bad-health days, but the model sees 88 days. Fixing such codes was
measured at about +0.005 F1.

## Where to start

The 15 rows with `first_batch = yes` are the risk factors that mattered most
in our experiments (age, sex, blood pressure, cholesterol, smoking, ...).
Do those first. Three rows are filled in as examples: **PHYSHLTH**,
**ALCDAY5** and **BPHIGH4** — look at them before starting.

Codebook: [CDC BRFSS 2015 codebook (PDF)](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf).
Search it for the column name (e.g. `SAS Variable Name: GENHLTH`); it lists
every value and its meaning.

## The columns

Filled by the script (do not edit — rerun the script instead):

| Column | Meaning |
|---|---|
| `column`, `index` | name and position in `x_train` |
| `label` | the question title from the codebook |
| `first_batch` | `yes` = one of the 15 to do first |
| `missing_pct` | % of rows that are empty |
| `n_distinct` | how many different values occur |
| `top_values` | the 5 most frequent values and their share |
| `detected_codes` | codes today's preprocessing already turns into "missing" |
| `kept_now` | `yes` if today's preprocessing uses the column at all |

Filled by hand, from the codebook:

| Column | What to write |
|---|---|
| `type` | `binary`, `categorical`, `ordinal`, `count`, `continuous`, `id`, `date` or `weight` |
| `special_codes` | every code that is not a plain value, with its meaning (`88 = none; 77 = don't know; 99 = refused`) |
| `decision` | `keep`, `recode`, `one-hot` or `drop` |
| `notes` | what the recode should do, or why you decided so |
| `checked_by` | your name |

Read by the code (`src/preprocessing.py`, option `codes=True` and the later
steps), so keep them machine-readable:

| Column | What to write |
|---|---|
| `nan_codes` | codes that mean don't know / refused / not on the scale, space separated (`77 99`); filled from the codebook labels |
| `zero_codes` | codes that mean none / never and become 0 (`88`) |
| `not_asked_fill` | what a blank means under skip logic (documentation; the rules themselves are `SKIP_RULES` in `src/preprocessing.py`) |
| `family` | duplicate family (all age columns are `Age`); one member is kept |
| `priority` | `A` do first, `B` next, `C` minor, `-` dropped |

**Status (4 Oct 2026):** all 321 rows are filled from a full pass over the
CDC 2015 codebook. The reasoning, findings and ablation plan are in
[`preprocessing_plan.pdf`](preprocessing_plan.pdf).

Rules of thumb for `decision`:

- **recode** when a code stands for a real value (88 = "none" -> 0) or the
  column mixes units (ALCDAY5: per week and per month in one column);
- **one-hot** when the numbers are only labels with no order (BPHIGH4:
  "yes", "only during pregnancy", "no", "borderline");
- **keep** when the values are already a proper number or an ordered scale
  and only "don't know / refused" (7/9, 77/99, ...) must become missing;
- **drop** for ids, dates, survey weights and columns asked to almost nobody.

You can edit the CSV in VS Code, LibreOffice or any spreadsheet program — just
save it as CSV again. Rerunning `python tools/make_feature_registry.py` keeps
everything filled by hand.
