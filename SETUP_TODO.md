# Author setup checklist

## Completed in clean-clone testing

- [x] Record baseline repository URLs, pinned commits and compatibility patches in `baseline_sources.json` and `patches/`.
- [x] Publish the `baseline-assets-v1` archive and checksum file; compare downloaded SHA-256 with the published archive digest.
- [x] Prepare a fresh checkout of AT, G2S, TR and RetroSim, including patch application.
- [x] Install the four exported baseline environments on the author's older CentOS Linux server (not a guarantee of cross-platform compatibility).
- [x] Install the selector in Python 3.10; use `constraints-centos7.txt` for the pandas prebuilt-wheel workaround on the tested server.
- [x] Configure local `*_PYTHON` paths and verify the four baseline assets; run real one-target baseline inference and produce a merged candidate pool.
- [x] Complete an offline MoA smoke test with synthetic responses and a recorded-response replay of one real Llama/GPT/Gemma sample. The replayed Top-10 SMILES matched the saved experiment output.

## Still to verify before claiming full reproduction or public-ready redistribution

- [ ] Validate live user-configured LLM/provider calls (model identifiers, authentication, response formats and current API behavior).
- [ ] Independently compare every released checkpoint, vocabulary, template file and RetroSim corpus against the paper's actual job artifacts; publish per-file hashes and provenance.
- [ ] Reproduce and verify the complete USPTO-50K experiment, including paper-specific settings and quantitative results.
- [ ] Test invalid-ID and other error-path cases with recorded responses; a successful recorded sample with no invalid selections does not cover these edge cases.
- [ ] Verify clean installation on additional Linux systems / GPU configurations.
- [ ] Select a code license and verify third-party source/data/checkpoint redistribution terms before switching repository visibility to public.

Original train/test datasets, experiment outputs, historical responses, plausibility analysis and temporary offline test scripts are intentionally excluded from this release. Do not commit local `.env` secrets.
