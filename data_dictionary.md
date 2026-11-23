# Data dictionary

## 1. Source and snapshot

The analysis uses four CSV tables from the [AIPolicyTracker Open Data Portal](https://aipolicytracker.org/open-data): `policies`, `obligations`, `jurisdictions` and `implementation`. The notebook setup records the source URL, licence label `CC BY 4.0`, access date `2026-11-02`, and the Kaggle folder hint `AIpolicy_data`. These are **configured metadata values in the notebook**; verify the actual download date, current licence terms, attribution instructions and dataset version before publication.

The dataset is dynamic. Row counts and column names must be taken from the final frozen download/run metadata, not copied from an earlier run. The earlier analysed snapshot described in the project materials contained 244 policy records, 117 obligation records, 249 jurisdiction records and 8 implementation records; these counts are not guaranteed for a later download.

## 2. Tables and analytical roles

| CSV table | Unit / key | Role in analysis | Important fields used by the notebook |
|---|---|---|---|
| `policies` | Policy instrument; `slug` | Primary unit of analysis, coding, corpus description and policy-level models | `slug`, `title`, `summary_plain`, `who_it_applies_to`, `jurisdiction`, `instrument_type`, `is_binding`, `source_url` and/or `url`, date fields such as `adopted_on`, `published_on`, `in_force_on`, `applies_from` |
| `obligations` | Obligation record; `slug`; linked by `policy` | Obligation categories and actor references; obligation text is aggregated to policy level for coding | `slug`, `policy`, `jurisdiction`, `title`, `summary`, `practical_action`, `is_binding`, `category` |
| `jurisdictions` | Jurisdiction; `slug` | Jurisdiction type and geographic metadata | `slug`, `jurisdiction_type`, `region`, `subregion` |
| `implementation` | Implementation record | Descriptive evidence about implementation stage | Available fields include `slug`, `kind`, `status`, `source_tier`, `published_on`, `confidence_level`; optional policy-link columns are detected if present |

The exact columns can vary across tracker releases. The notebook's data dictionary and quality-check output (`t01_data_dictionary.csv`, `t02_data_quality.csv`) should be retained with each run.

## 3. Joins and text construction

- Jurisdiction metadata is joined to policies using `policies.jurisdiction = jurisdictions.slug`; the merge is validated as many-to-one.
- Obligations are grouped by `obligations.policy`, with the linked policy slug used as the join key. The notebook aggregates the obligation title, summary and practical-action text and counts the number of linked obligations.
- Policy coding text is constructed by concatenating policy `title`, `summary_plain`, `who_it_applies_to` and the aggregated linked-obligation text.
- Obligation actor analysis uses obligation text and the obligation `category` field.

The original four loaded dataframes are copied to `pol_raw`, `obl_raw`, `jur_raw` and `imp_raw` before transformations as read-only audit copies during the notebook session.

## 4. Derived policy variables

| Variable | Definition |
|---|---|
| `format` | Mapping from `instrument_type`: `act`, `regulation`, `bill`, `executive_order` → `statute`; `strategy` → `strategy`; `policy` → `policy`; `guidance`, `framework`, `standard`, `code_of_practice` → `soft_guidance`; `consultation` → `consultation`; unmapped values → `other`. |
| `format3` | Collapses `strategy`, `policy` and `consultation` into `strategy_policy`; other format categories are retained. Used to reduce sparse categories in regression. |
| `tier` | Derived from `jurisdiction_type`; `territory` and `region` are recoded as `state`; country, state and supranational remain separate. |
| `is_binding` | Tracker `is_binding` field, with missing values filled as `False` and cast to Boolean. This is a tracker-derived classification and should be checked against official instruments for key cases. |
| `law_type` | Label `binding (hard law)` when `is_binding` is true; otherwise `non-binding (soft law)`. |
| `obl_text` | Concatenated obligation title, summary and practical action, aggregated by policy. |
| `text` | Concatenated policy title, summary_plain, who_it_applies_to and linked `obl_text`. |
| `n_obligations` | Number of obligation records linked to each policy. |

## 5. Coder-derived variables

For each coding procedure (`A_`, `B_`) and the reconciled working coding (no prefix):

- `grp_<group>`: group-level score 0–3 for each of five stakeholder groups.
- `remedy`: document-level maximum remedy score 0–3.
- `public_enforcement`: binary public-enforcement reference.
- `duty_holder`: binary actor-plus-duty expression in the same sentence.
- `oversight`: binary oversight/enforcement lexicon match.
- `impl_support`: binary implementation-support lexicon match.
- `n_groups`: number of stakeholder groups with score greater than 0.
- `inclusion`: 1 if at least one stakeholder group has score greater than 0.
- `n_groups_active`: number of groups with score 2 or 3.
- `enforceable_remedy`: 1 if document remedy score equals 3.
- `procedural_protection`: 1 if document remedy score equals 2.
- `procedural_or_enforceable`: 1 if document remedy score is at least 2.
- `f_right_protection`: 1 if any stakeholder group has score at least 2.
- `f_duty_holder`, `f_oversight_enforcement`, `f_complaint_remedy`, `f_implementation_support`: four additional binary accountability features; the first feature is `f_right_protection`.
- `accountability_index`: sum of the five accountability features, 0–5.

The group-level stakeholder protection score and document-level remedy score are different constructs and must not be conflated.

## 6. Obligation and jurisdiction variables

The obligation table receives actor flags for provider/developer, deployer/user organisation, importer/distributor, public authority and general-purpose/frontier model references. These categories are non-exclusive. The notebook also identifies operational/technical obligation categories using its `OPERATIONAL` category list; consult the executed notebook cell and saved tables for the exact list used in a given version.

Jurisdiction-level profiles include policy count, obligation count, percentage binding, mean number of stakeholder groups recognised per policy, percentage recognising any group, percentage with a strong remedy, and mean accountability index. Only jurisdictions with at least `MIN_DOCS = 3` policies appear in the principal jurisdiction comparison table. Profiles are descriptive, not jurisdiction rankings.

The implementation table is summarised by record kind, status and confidence where those fields exist. A record status of `unverified` means publication was not confirmed in the tracker; it does not prove the measure does not exist.

## 7. Data quality and missingness

The notebook reports duplicate slugs, missingness in selected text fields, unmatched obligation-to-policy keys and unmatched policy-to-jurisdiction keys. Preserve these checks for each data snapshot. Missing text can reduce keyword detection and should be discussed as a measurement limitation.

## 8. Run-specific record

For every analysis run, archive:

- the four source CSV files or a lawful, reproducible access reference and file hashes;
- `t01_data_dictionary.csv` and `t02_data_quality.csv`;
- `run_metadata.csv` and `manifest.csv`;
- the notebook version/commit and execution date;
- the completed human-validation/adjudication files, if any.
