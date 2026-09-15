#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# LLMServingSim-side patches to the Chakra converter that are not yet
# upstream in casys-kaist/chakra (see scripts/patches/*.patch). Idempotent.
for p in "${REPO_ROOT}"/scripts/patches/chakra-*.patch; do
  [ -f "$p" ] || continue
  git -C "${REPO_ROOT}/astra-sim/extern/graph_frontend/chakra" apply --check "$p" 2>/dev/null \
    && git -C "${REPO_ROOT}/astra-sim/extern/graph_frontend/chakra" apply "$p"
done

# Intall Chakra (Use Chakra fork in ASTRA-Sim repo).
(
cd ${REPO_ROOT}/astra-sim/extern/graph_frontend/chakra
pip3 install .
)

# Compile ASTRA-sim with analytical backend model
(
cd ${REPO_ROOT}/astra-sim
bash ./build/astra_analytical/build.sh
)

# Compile ASTRA-sim with ns3 backend model
# (
# cd ${REPO_ROOT}/astra-sim
# bash ./build/astra_ns3/build.sh
# )
