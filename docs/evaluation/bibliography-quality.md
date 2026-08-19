# Bibliography Quality

Bibliography quality evaluates extracted bibliography entries against curated ground truth references.

The retained evaluation scripts compare GROBID output and Kreuzberg plus regex output. This evaluation level is independent from table reconstruction quality: it scores whether the bibliography extractor recovered the correct reference entries before any table-reference matching is attempted.

Useful measures include entry count agreement, exact or fuzzy text similarity, normal accuracy, and F1-style precision/recall depending on the curated ground truth available.

Bibliography evaluation should read extracted bibliography artifacts and write metrics separately. It should not modify `references/bibliography.json`, prediction CSV files, or resolved CSV files.
