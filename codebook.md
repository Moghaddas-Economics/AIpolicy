# Codebook

## 1. Purpose and unit of analysis

This codebook documents the operational rules implemented in `notebooks/aipolicy_analysis.ipynb` for the study **Stakeholder Power and Governance Asymmetries in AI Policy: from Recognition to Enforceable Accountability**.

The primary unit of analysis is the **policy instrument** (one row in the AIPolicyTracker `policies` table). Obligation records are linked to policy instruments through `obligations.policy == policies.slug` and are also analysed as obligation-level records for actor references. Jurisdiction and implementation tables provide contextual and descriptive information.

The coded text is assembled from policy `title`, `summary_plain`, `who_it_applies_to`, and linked obligation `title`, `summary`, and `practical_action`. It is tracker-derived text and is **not necessarily the complete official legal instrument**. Absence of a keyword or mechanism in this text must not be interpreted as proof that the full instrument lacks it.

## 2. Conceptual sequence and research questions

The analytical framework is:

**Recognition → Protection → Accountability → Implementation**

- **RQ1:** How are different stakeholder groups represented across AI-governance policies and jurisdictions?
- **RQ2:** To what extent do AI policies translate stakeholder recognition into clearly defined rights, protections and obligations?
- **RQ3:** What accountability mechanisms (oversight, enforcement and remedies) are established to support compliance with AI-policy obligations?
- **RQ4:** How do stakeholder protection and enforceable accountability differ across jurisdictions and policy-implementation stages?

The stages are not assumed to be present in every instrument. The framework is used to identify which links are visible in the available tracker text and records.

## 3. Stakeholder groups and group-level score

Each of the five stakeholder groups receives a score from 0 to 3. A group is assigned a score only when the group term and the relevant protection/remedy expression occur in the **same sentence**. The maximum qualifying sentence-level score is retained for that group across the document.

| Group | Examples in the implemented lexicon |
|---|---|
| Consumers / individuals | consumers, individuals, data subjects, natural persons, citizens, end-users, residents, people |
| Workers | workers, employees, job applicants, candidates, labour, staff |
| Children | children, child, minors, youth, young people; Coder B also includes juvenile/young person |
| Vulnerable groups | vulnerable persons/people/groups, persons/people with disabilities, older persons, elderly people, patients, marginalised/marginalized groups |
| Civil society | civil society, public-interest groups, NGOs, affected communities, community representatives, public consultation |

### Group-level scale

| Score | Label | Operational meaning |
|---:|---|---|
| 0 | Absent | No group mention detected by the relevant coder. |
| 1 | Passive recognition | Group is mentioned, but no qualifying procedural safeguard or explicit individual right/remedy is linked in the same sentence. |
| 2 | Procedural protection | The group is mentioned in the same sentence as a procedural safeguard, such as notice, explanation, complaint, appeal, contestation, human review/intervention, opt-out or a review procedure. |
| 3 | Explicit individual right/remedy | The group is mentioned in the same sentence as an explicit individual right/remedy, such as a right to compensation, damages, redress, a legal/judicial remedy or a right of action. |

The same-sentence rule is a deliberate conservative linkage rule. It does not establish that the legal right is enforceable in every jurisdiction; official-source review is needed for legal interpretation.

## 4. Document-level remedy strength

A separate document-level score records the highest remedy tier found in any sentence, whether or not a stakeholder group appears in that sentence.

| Score | Label | Implemented interpretation |
|---:|---|---|
| 0 | None detected | No remedy/safeguard term in the implemented lexicons. |
| 1 | Symbolic / aspirational | Principles, encouragement, promotion, voluntary language, should, ethics, trustworthy/responsible AI, guidance or best practice. |
| 2 | Procedural | Appeal, contest, challenge, complaint/grievance, human review/intervention, notice/notification, explanation, specified access/objection/erasure/opt-out rights or review procedure. |
| 3 | Explicit individual right/remedy | Explicit right to compensation, damages, redress or a remedy; right of action/private right of action; claim/seek/obtain compensation or redress; judicial/legal remedy or remedy for an individual. |

Priority is hierarchical: an explicit-right pattern takes precedence over a procedural pattern, which takes precedence over a symbolic pattern. Coder B applies a modal-language rule: where soft/aspirational modal wording is present and the sentence does not qualify for tier 2 or above, the score is capped at 1.

**Important distinction:** document-level remedy strength is not automatically a stakeholder-linked remedy. The paper must report the group-linked scores separately from the maximum document-level score.

## 5. Public enforcement

A separate binary indicator records references to fines, financial/administrative penalties, sanctions, administrative measures or prosecution. This indicator represents **public enforcement references**, not an individual right to compensation or redress. It is not coded as an individual remedy merely because a regulator can impose a sanction.

## 6. Accountability features

Five binary features are derived for each policy document:

| Feature | Value 1 when the implemented rule detects… |
|---|---|
| Right / protection | At least one stakeholder group has a group-level score of 2 or 3. |
| Duty-holder | An actor and an obligation/duty expression occur in the same sentence. |
| Oversight & enforcement | An audit, supervision, monitoring, oversight, inspection, competent authority, market-surveillance, conformity-assessment, regulator/enforcement-authority or public-enforcement term is detected. |
| Complaint & remedy | Document-level remedy score is at least 2. |
| Implementation support | A standard, guideline, code of practice/conduct, template, procedure, implementing act/measure or technical specification is detected. |

The **accountability index** is the unweighted sum of these five binary features, ranging from 0 to 5. It is an exploratory descriptive index, not a validated scale. Some components are keyword-derived and can produce false positives or miss context; results should be interpreted accordingly.

## 7. Actor categories for RQ4

Actor references are coded in obligation text and categories can overlap; totals therefore need not sum to the number of obligations.

- Provider / developer: providers, developers, manufacturers, vendors, suppliers.
- Deployer / user organisation: deployers, operators, organisations/organizations, businesses, companies, employers, controllers, processors.
- Importer / distributor: importers, distributors, authorised/authorized representatives.
- Public authority: public authorities/bodies/sector, government, agencies, regulators, ministries, authorities.
- General-purpose / frontier model: general-purpose AI models, foundation models, frontier models. The notebook explicitly excludes generic terms such as “systemic risk” or “high-impact” as actor labels.

The duty-holder indicator is stricter than an actor reference: the actor and an obligation expression must appear in the same sentence.

## 8. Coder A and Coder B

Both coders are deterministic rule-based regular-expression procedures; neither is a human coder, machine-learning model or generative-AI classifier.

- **Coder A (strict/conservative):** uses whole-word/boundary-based patterns and a tiered remedy lexicon.
- **Coder B (broader):** uses stem/substring-oriented patterns and a modal-verb rule that caps aspirational language at the symbolic tier unless a procedural mechanism is detected.

The notebook calculates Cohen's kappa, raw agreement and bootstrap confidence intervals for A-versus-B comparisons. Weighted kappa is used for ordinal 0–3 scores and the accountability index; unweighted kappa is used for binary indicators. These statistics must be described as **inter-method agreement**, not human inter-coder reliability or construct validity.

For the principal working analysis, the notebook reconciles each coded score using the lower of A and B. This minimum rule is a conservative, pre-specified working specification and **not** a human-adjudicated ground truth. The sensitivity analysis reports headline results under A, B and the reconciled coding.

## 9. Human validation and adjudication

The notebook exports `human_validation_template.csv` with a reproducible stratified sample (up to 60 policy documents), stratified by the reconciled document-level remedy category where possible. The sample is a review worksheet, not evidence that validation has been performed.

Reviewers should inspect the linked official policy source and independently record:

- whether the official source was checked;
- remedy score (0–3) for each reviewer;
- any-stakeholder inclusion (0/1) for each reviewer;
- adjudicated scores after resolving disagreements;
- notes and a short supporting quotation/reference.

Do not report human validation as completed until the worksheet has been populated, source checks have been documented, and reviewer agreement/adjudication has been analysed. If two reviewers code the same cases independently, human-human weighted kappa can be reported for the ordinal remedy score, with sample size and confidence interval where calculated. Automated-versus-human comparison should also be reported if implemented.

The optional `human_adjudication.csv` input expected by the notebook must contain unique `slug`, `adjudicated_remedy` (0–3) and `adjudicated_inclusion` (0/1) columns. It overrides document-level remedy and inclusion values where supplied. It does not automatically replace every group-level code.

## 10. Interpretation boundaries

1. Keyword detection in tracker-derived text is not a legal opinion and does not establish enforceability.
2. Missing tracker text is not evidence that a protection is absent from the full instrument.
3. Public sanctions are not individual remedies.
4. A document-level remedy is not necessarily linked to a named stakeholder.
5. Actor categories overlap.
6. The small implementation-record table is descriptive only and cannot establish implementation effectiveness.
7. Regression models are exploratory associations, not causal estimates; models are withheld when event counts, convergence or separation checks are not satisfactory.
