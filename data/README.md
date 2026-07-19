# Trace Data

## Vendored Corpus

The Phase 4B corpus contains six unchanged WfFormat 1.5 execution traces from
the official `wfcommons/WfInstances` repository. They were retrieved on
2026-07-19 under the upstream GNU Lesser General Public License v3.0.

| Family | Size role | Tasks | Edges | Files | Local file |
| --- | --- | ---: | ---: | ---: | --- |
| Montage | small benchmark | 58 | 114 | 111 | `raw/wfcommons/montage-chameleon-2mass-005d-001.json` |
| Montage | medium held-out | 310 | 798 | 471 | `raw/wfcommons/montage/montage-chameleon-2mass-015d-001.json` |
| Epigenomics | small benchmark | 73 | 88 | 94 | `raw/wfcommons/epigenomics/epigenomics-chameleon-hep-1seq-50k-001.json` |
| Epigenomics | medium held-out | 445 | 550 | 558 | `raw/wfcommons/epigenomics/epigenomics-chameleon-hep-3seq-50k-001.json` |
| Seismology | small benchmark | 101 | 100 | 304 | `raw/wfcommons/seismology/seismology-chameleon-100p-001.json` |
| Seismology | medium held-out | 201 | 200 | 604 | `raw/wfcommons/seismology/seismology-chameleon-200p-001.json` |

SHA-256 checksums:

```text
5795e0ab9e13bb7d50d046796bcbc8ec0a884eba0512a95222bbd557bc6d0b65  montage-chameleon-2mass-005d-001.json
d48ebd2e26e34a2c086cb4284b2fccc28a8708c4fcb5e323cce6e2280a336889  montage-chameleon-2mass-015d-001.json
8a64ae6acab57415972207de056d66d4cdea973b4a8f8fdf8200268cf269273e  epigenomics-chameleon-hep-1seq-50k-001.json
e10731e3cc1e116327a5a127cd528981b5ca71e1a0d5681fa8e3a6bf545c0850  epigenomics-chameleon-hep-3seq-50k-001.json
99c0426009e1e2a0316c53da45ce9247ffaeee7d2e0fad6b4de96e796b5aaad4  seismology-chameleon-100p-001.json
91ce08abe3d980184a1f000967ed264eeba0475b175a9052334e63038d2bd549  seismology-chameleon-200p-001.json
```

Official raw URLs, checksums, and expected object counts are serialized in
`configs/workflow_benchmark.json`. The loader verifies them before every
benchmark, so experiments run offline after the one-time download and detect
accidental trace changes.

These files provide real DAG structure, observed runtimes, and file metadata
from executions on Chameleon. They do not contain counterfactual measurements
for this project's `Compute-Fast`, `Balanced`, and `IO-Fast` workers. Worker
heterogeneity, network bandwidth, workflow arrivals, and runtime noise remain
documented simulation assumptions.
