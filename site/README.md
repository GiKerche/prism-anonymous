# Interactive experiment explorer

The static site in `docs/` loads its data in the browser. It has no backend,
analytics, external scripts, hosted model connection, or runtime credentials.
All application links and data paths are relative, so the files can be hosted
under a project subdirectory or an anonymous artifact mirror.

## Build and verify

From the repository root:

```bash
python site/build_data.py
python site/verify_data.py
python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

The builder reads the deposited scientific exports and frozen vectors. It
creates a compact numerical catalog, the final-test prompts and tool schemas,
the two-component display projections, and a copy of the anonymous paper.
No experiment is rerun. Rebuild the paper before running the builder if its
source changes.

The 33 compressed response shards are deposited in `docs/data/responses`.
They contain 66,304 recorded responses, exported by matching scenario, language,
context, condition and seed to the final scientific endpoints. They include
the main source interventions, language-specific interventions, Arabic,
transported receiver interventions, and independently selected receiver-native
comparators. Each response retains its content hash. The shard manifest contains
file hashes that are checked by the browser before decompression.

Each JSON or gzip data file also has a generated `.js` companion. These static
payloads preserve its exact bytes in base64 and load as classic scripts. This
supports anonymous hosts that assign pages an opaque origin and block module
scripts and `fetch`. No data request leaves the hosting mirror. Regenerate the
companions with `python site/build_browser_payloads.py` after changing any data
file; `build_data.py` also runs this step. `verify_data.py` checks every companion
against its underlying file before checking the scientific endpoints.

One generated personal name in one response was replaced for anonymous review.
That record explicitly marks the display change, retains the original response
hash, and adds a separate display hash. Its labels, condition and statistics are
unchanged. All other displayed response texts retain their original hashes.

`verify_data.py` checks every response's display/content hash, every deposited
endpoint label, and all primary rates against the numerical exports. It also
checks zero-dose aliases and the independently selected receiver comparators.

## Interpretation

- A single displayed response is an example. Aggregate rates and paired effects
  use the specified test group, independently of the example filter.
- Missing judgments stay missing. They are not converted into nonviolations.
- Positive doses use the ethical context; negative doses use the neutral context.
  Both contain the same harmful user request. Each intervention is paired with
  the matching context's baseline.
- Native and shadow share BF16 activation space; only their response-label
  sources differ. Model transport uses the source's pooled native direction.
- The dose slider displays the saved Validation II grid's structural call proxy
  and BFCL accuracy. It does not interpolate final semantic effects at untested
  doses. An admissible dose can still differ from the selected dose.
- The geometric plot is a two-dimensional projection of centered unit vectors.
  Its display coordinates do not replace full-space cosine measurements.
- LUMEN records proposed calls. SODA's executed trajectories remain a separate
  experiment described in the paper and the reproducibility artifacts.

## Hosting

Serve only `docs/`, or use the identical static files in the `gh-pages` branch.
The numerical fitting artifacts in the repository root are not copied into the
Pages site. A current browser supporting `DecompressionStream('gzip')` is needed
for response browsing. The PDF remains available without JavaScript.
On Anonymous GitHub, use **two separate mirrors**. The full scientific
repository must mirror the **`main` branch at its root**, with downloads
enabled. Its README explains how to restore the larger numerical artifacts
from parts that respect the host's file-size limit. A mirror of only `docs/`
or `gh-pages` does not provide the full reproducibility artifact.

For the interactive website, mirror **`gh-pages`** and enable **GitHub Pages**.
Use its **Website** link for the interactive view; an explicit `index.html`
path also works when the host does not resolve the directory index.
After updating the source,
refresh the mirror snapshot in the owner's dashboard. Keep the `.js` data
companions when copying the site. On hosts that block file downloads, **Export
pair** presents the exact JSON for copying instead.

The deposited HTML, data, PDF, citation file and new Git history omit author
identity. A hosting account or repository URL may itself reveal ownership;
use an anonymous mirror when sharing the artifact for anonymous review.
