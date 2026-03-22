# arXiv Posting Sequence

## Pre-posting
- [ ] Final read-through of main paper
- [ ] History squash (optional — the `ac14d0c` "ALL four datasets" commit)
- [ ] Make repo public at https://github.com/mojoatomic/stts

## Posting order

1. **Make repo public** at github.com/mojoatomic/stts

2. **Post main STTS paper** to arXiv (cs.LG, eess.SY, cs.DB)
   - Get arXiv ID (e.g., `2603.XXXXX`)

3. **Update companion paper** `[^stts]` footnote with main paper arXiv ID:
   ```
   [^stts]: Fennell, D. (2026). State Topology and Trajectory Storage...arXiv:2603.XXXXX
   ```

4. **Post companion paper** to arXiv (astro-ph.EP, cs.LG)
   - Get arXiv ID (e.g., `2603.YYYYY`)

5. **Update main paper** `[^stts_orbital]` footnote with companion arXiv ID:
   ```
   [^stts_orbital]: Fennell, D. (2026). STTS-Orbital...arXiv:2603.YYYYY
   ```

6. **Update both papers** on arXiv same day (arXiv allows same-day replacements)

## Post-posting

- Notify CNEOS team at JPL (Paul Chodas)
- Notify ESA Planetary Defence Office (Richard Moissl)
- Notify Scout/Sentry development teams
