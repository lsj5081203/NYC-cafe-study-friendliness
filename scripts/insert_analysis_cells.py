"""
One-shot script: insert two analysis markdown cells into notebook 04
immediately before the '## 6. Export Results' cell.

Run from the project root:
    python3 scripts/insert_analysis_cells.py
"""

import json
import copy

NOTEBOOK_PATH = "notebooks/04_cafe_scoring.ipynb"

CELL_A = {
    "cell_type": "markdown",
    "id": "cell-analysis-score-interpretation",
    "metadata": {
        "id": "cell-analysis-score-interpretation"
    },
    "source": [
        "## Score Interpretation\n",
        "\n",
        "### Why all seven cafes landed in the \"Fair\" band (35\u201355/100)\n",
        "\n",
        "The narrow clustering is **structurally expected**, not a measurement failure. Three\n",
        "independent forces conspire to anchor every urban cafe near the middle of the scale.\n",
        "\n",
        "#### 1. Domain gap between UrbanSound8K and cafe ambience\n",
        "\n",
        "UrbanSound8K\u2019s 10 classes were curated for *outdoor* urban taxonomy\n",
        "(street, construction, traffic, nature). A cafe recording contains sounds that\n",
        "exist **nowhere** in that vocabulary: espresso steam wands, cutlery clatter,\n",
        "overlapping conversation, ambient background music. The CNN has no choice but to\n",
        "force-classify every 4-second window into the nearest available bucket.\n",
        "\n",
        "Empirically, these forced misclassifications tend to land on classes with\n",
        "**intermediate to high distraction weights** \u2014 `siren` (0.85), `children_playing`\n",
        "(0.50), `street_music` (0.40), `car_horn` (0.90) \u2014 rather than on the genuinely\n",
        "quiet classes such as `air_conditioner` (0.10) or `engine_idling` (0.15).\n",
        "This is acoustically plausible: conversation and cutlery share spectral and\n",
        "temporal characteristics with those classes far more than with a steady\n",
        "low-frequency hum.\n",
        "\n",
        "**Working through the math.** To reach the observed acoustic scores of 39\u201356,\n",
        "the per-recording weighted distraction must fall between 0.44 and 0.61. A\n",
        "representative mixture \u2014 40% windows \u2192 `street_music` (0.40), 30% \u2192 `siren`\n",
        "(0.85), 20% \u2192 `children_playing` (0.50), 10% \u2192 `engine_idling` (0.15) \u2014\n",
        "yields a weighted distraction of 0.525 and an acoustic score of **47.5**,\n",
        "exactly the neighbourhood where every cafe landed.\n",
        "\n",
        "#### 2. Score formula ceiling imposed by Wi-Fi cap\n",
        "\n",
        "The final formula is:\n",
        "\n",
        "```\n",
        "final = 0.9 \u00d7 acoustic + 0.1 \u00d7 wifi_bonus \u2212 0.05 \u00d7 eatery_penalty\n",
        "```\n",
        "\n",
        "With acoustic scores bounded at ~40\u201356, the acoustic term alone contributes\n",
        "36\u201350 to the final score. The Wi-Fi bonus is capped at **+10 points** (10\n",
        "hotspots \u2192 wifi_bonus = 100 \u2192 weight 0.10 \u2192 +10 pts). Even at maximum Wi-Fi\n",
        "saturation, no cafe in this cohort could exceed ~66 \u2014 and that would require\n",
        "the best acoustic score *and* 10+ hotspots simultaneously. Joe Coffee (6\n",
        "hotspots) is closest to that ceiling but its per-recording acoustic scores\n",
        "(40\u201341) keep it firmly in Fair territory.\n",
        "\n",
        "#### 3. Eatery API returned zero data\n",
        "\n",
        "All seven cafes show `eatery_count = 0` due to a **HTTP 403 Forbidden** error\n",
        "from the NYC Open Data eateries endpoint. This silently removes the eatery\n",
        "penalty term. In dense commercial areas like LIC (Jacx & Co, Joe Coffee,\n",
        "Utopia Bagel), the true eatery count within 500 m is likely 30\u201360+, which\n",
        "would subtract up to 3 points from those scores. Relative rankings are not\n",
        "materially affected, but absolute scores are mildly inflated for high-density\n",
        "locations.\n",
        "\n",
        "#### What a \"Good\" or \"Excellent\" score would require\n",
        "\n",
        "| Target grade | Required acoustic score | What that implies for the CNN |\n",
        "|---|---|---|\n",
        "| Good (55\u201374) | \u2265 56 | Consistently classifies windows as `air_conditioner` (0.10) or `engine_idling` (0.15) \u2014 i.e., genuinely quiet, steady-state backgrounds |\n",
        "| Excellent (\u226575) | \u2265 75 | Virtually all windows must map to the two quietest classes simultaneously |\n",
        "\n",
        "Reaching \"Good\" from real NYC cafe recordings is achievable only with\n",
        "(a) a domain-adapted model trained on actual cafe ambience, or\n",
        "(b) deliberate recalibration of `DISTRACTION_WEIGHTS` to reflect cafe-context\n",
        "expectations rather than outdoor urban severity.\n",
        "\n",
        "#### The standout result: Blank Street Cafe (inside)\n",
        "\n",
        "Blank Street Cafe\u2019s inside recording reached an acoustic score of **55.51** \u2014\n",
        "the highest in the dataset by a 4.7-point margin over the next best recording.\n",
        "Its outside recording scored only 40.34, a **15.2-point inside/outside gap**\n",
        "that is the largest asymmetry across all seven venues. This is the most\n",
        "interpretable single finding: interior acoustic treatment (smaller room, soft\n",
        "furnishings, absorption of street noise) produces a sound environment\n",
        "detectably different from its own sidewalk, even through the domain gap."
    ]
}

CELL_B = {
    "cell_type": "markdown",
    "id": "cell-analysis-limitations",
    "metadata": {
        "id": "cell-analysis-limitations"
    },
    "source": [
        "## Limitations and Recommendations\n",
        "\n",
        "### Known limitations of this scoring run\n",
        "\n",
        "#### L1 \u2014 Eatery API failure\n",
        "The NYC Open Data eateries endpoint (`/resource/8792-ebcp.json`) returned\n",
        "**HTTP 403 Forbidden** for all requests, setting every `eatery_count` to 0.\n",
        "The eatery penalty term (`\u22120.05 \u00d7 eatery_penalty`) is silently zeroed out.\n",
        "**Fix:** re-authenticate with a valid Socrata app token, or substitute the\n",
        "DOHMH restaurant inspection dataset which uses a different endpoint URL.\n",
        "Correcting this will lower absolute scores slightly (up to \u22123 pts in dense\n",
        "neighbourhoods) but will not change the relative cafe ranking unless one\n",
        "location is dramatically more isolated than the others.\n",
        "\n",
        "#### L2 \u2014 UrbanSound8K class vocabulary vs. cafe ambience (domain gap)\n",
        "The classifier was trained exclusively on 10 outdoor urban classes; none\n",
        "represents normal cafe ambience. Every window prediction is a forced choice.\n",
        "Scores should be interpreted as a **relative ranking** within this cohort,\n",
        "not as absolute measures of acoustic comfort. The \"Fair\" label does not imply\n",
        "these are unpleasant study environments \u2014 it reflects the limits of applying\n",
        "an outdoor sound taxonomy to indoor settings.\n",
        "\n",
        "#### L3 \u2014 Single-visit, single-recording per setting\n",
        "Each location was recorded once. Cafe noise levels vary significantly by time\n",
        "of day and day of week. A single recording may not represent a typical study\n",
        "session and should be treated as a snapshot rather than a stable estimate.\n",
        "\n",
        "### Recommendations for future iterations\n",
        "\n",
        "**R1 \u2014 Recalibrate distraction weights for cafe context.**\n",
        "The most impactful single change is updating `DISTRACTION_WEIGHTS` in\n",
        "`src/scoring.py` to reflect what is actually distracting *during studying*,\n",
        "rather than general outdoor severity. For example, `street_music` (currently\n",
        "0.40) is often a *positive* ambient signal in cafes; lowering it to 0.20 would\n",
        "immediately differentiate locations where that class dominates from those\n",
        "dominated by `siren` (0.85) or `car_horn` (0.90).\n",
        "\n",
        "**R2 \u2014 Use confidence-weighted prediction scores.**\n",
        "Instead of taking the argmax class per window, use the CNN\u2019s softmax\n",
        "probability vector directly:\n",
        "\n",
        "```python\n",
        "# Expected distraction per window (soft weighting)\n",
        "distraction = sum(prob[c] * DISTRACTION_WEIGHTS[c] for c in range(10))\n",
        "```\n",
        "\n",
        "This prevents a window with 52% `siren` confidence and 48% `street_music`\n",
        "confidence from being penalised as harshly as an unambiguous siren detection,\n",
        "and is more honest about model uncertainty on in-domain cafe sounds.\n",
        "\n",
        "**R3 \u2014 Collect a small cafe-ambience labeled dataset.**\n",
        "Even 100\u2013200 labeled 4-second clips covering espresso machines, cutlery,\n",
        "conversation, and background music would enable fine-tuning the CNN\u2019s\n",
        "final classifier layer. This is a low-data-cost intervention that could\n",
        "raise effective accuracy on cafe inference substantially and break the\n",
        "forced-choice constraint.\n",
        "\n",
        "**R4 \u2014 Fix the eatery API and refine the spatial radius.**\n",
        "The current 500 m haversine radius is coarse for NYC\u2019s dense street grid.\n",
        "A 200\u2013300 m radius for eateries would better capture walking-distance\n",
        "competition for study seats. Combining a tighter eatery radius with a\n",
        "wider Wi-Fi radius (hotspot availability at 500 m is genuinely relevant)\n",
        "would improve spatial signal quality without code changes beyond passing\n",
        "different `radius_m` arguments to `build_spatial_features()`."
    ]
}


def insert_cells():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the index of the Export Results markdown cell
    insert_idx = None
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "## 6. Export Results" in src:
            insert_idx = i
            break

    if insert_idx is None:
        raise ValueError("Could not find '## 6. Export Results' cell in notebook.")

    # Guard against double-insertion
    existing_ids = {c.get("id") for c in nb["cells"]}
    if "cell-analysis-score-interpretation" in existing_ids:
        print("Analysis cells already present — nothing to do.")
        return

    nb["cells"].insert(insert_idx, copy.deepcopy(CELL_B))
    nb["cells"].insert(insert_idx, copy.deepcopy(CELL_A))

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Inserted 2 analysis cells before cell index {insert_idx} "
          f"('## 6. Export Results').")
    print(f"Notebook now has {len(nb['cells'])} cells.")


if __name__ == "__main__":
    insert_cells()
