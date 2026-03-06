# Changelog

## [1.0.4](https://github.com/sam-dumont/musicgen-api/compare/v1.0.3...v1.0.4) (2026-03-06)


### Bug Fixes

* trim conditioning echo from generate_continuation() to eliminate double beats at segment boundaries ([e7fde3b](https://github.com/sam-dumont/musicgen-api/commit/e7fde3b256b90df7ae7d71a6323752a2c448960c))
* only tag Docker image as 'latest' on release tags, not every push to main ([e7fde3b](https://github.com/sam-dumont/musicgen-api/commit/e7fde3b256b90df7ae7d71a6323752a2c448960c))


### Dependencies

* upgrade torch 2.1.0→2.8.0, transformers→4.57.6, fix 25 dependabot alerts ([21c4833](https://github.com/sam-dumont/musicgen-api/commit/21c48337313ec4f1ebf2a9a49a5da894a96dc632))
* override numba>=0.59.0 and llvmlite>=0.42.0 for Python 3.12 compatibility ([e7fde3b](https://github.com/sam-dumont/musicgen-api/commit/e7fde3b256b90df7ae7d71a6323752a2c448960c))

## [1.0.3](https://github.com/sam-dumont/musicgen-api/compare/v1.0.2...v1.0.3) (2026-03-06)


### Bug Fixes

* remove transition bridge causing audio artifacts at scene boundaries ([#14](https://github.com/sam-dumont/musicgen-api/issues/14)) ([166089b](https://github.com/sam-dumont/musicgen-api/commit/166089b899d66ecfdf28bba89d93b1b470507d67))

## [1.0.2](https://github.com/sam-dumont/musicgen-api/compare/v1.0.1...v1.0.2) (2026-03-05)


### Bug Fixes

* prevent destructive tiny tail segments in soundtrack generation ([#12](https://github.com/sam-dumont/musicgen-api/issues/12)) ([43fa8f6](https://github.com/sam-dumont/musicgen-api/commit/43fa8f6752fb69611b42b04cfda51e10a41db59c))

## [1.0.1](https://github.com/sam-dumont/musicgen-api/compare/v1.0.0...v1.0.1) (2026-03-05)


### Bug Fixes

* prevent assertion error in sliding window generation ([#10](https://github.com/sam-dumont/musicgen-api/issues/10)) ([d5fa094](https://github.com/sam-dumont/musicgen-api/commit/d5fa0944dfd6ac23920ffbe7916a56dd03d93975))
