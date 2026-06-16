This repo builds a Python package (vllm-xpu-kernels) that registers custom PyTorch XPU (Intel GPU) ops used by vLLM.

Development happens inside the preconfigured Docker container:

docker exec -it -w $(pwd) hans-gpt-oss-ww17-fresh /bin/bash

The container already contains:

oneAPI toolchain

XPU-enabled PyTorch

Required compilers

Runtime libraries

Do all build + runtime work inside the container.

Entering the Dev Environment

From your host machine:

docker exec -it -w $(pwd) hans-gpt-oss-ww17-fresh /bin/bash

This:

Does NOT create a new container

Enters the running container

Sets working directory to your current host path (must be volume-mounted)

oneAPI Environment

The container may or may not auto-source oneAPI.

If you see errors like:

libccl.so.2: cannot open shared object file

missing icx / icpx

SYCL runtime errors

Run:

source /opt/intel/oneapi/setvars.sh
Python Environment

If the container already includes the correct Python environment, just activate it.

Otherwise create a local venv:

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

Supported: Python 3.9–3.13
Recommended: 3.12

Install Dependencies
pip install -r requirements.txt

If PyTorch needs reinstalling:

pip install --extra-index-url=https://download.pytorch.org/whl/xpu torch
Editable Build (Kernel Iteration Loop)

For C++/SYCL changes:

pip install --extra-index-url=https://download.pytorch.org/whl/xpu \
  --no-build-isolation -e . -v

Build artifacts:

build/temp/

vllm_xpu_kernels/*.so

Verify editable install:

pip show vllm-xpu-kernels | sed -n '1,120p'
oneDNN Dependency (IMPORTANT)

This repo builds oneDNN from source as a CMake subproject.

Required Location

The oneDNN checkout must exist at:

/data/josephku/oneDNN

cmake/Modules/FindoneDNN.cmake force-sets:

ONEDNN_ROOT=/data/josephku/oneDNN

The build does:

ADD_SUBDIRECTORY(${ONEDNN_ROOT} oneDNN EXCLUDE_FROM_ALL)

Experimental flags enabled:

DNNL_EXPERIMENTAL_GROUPED_MEMORY=ON

DNNL_EXPERIMENTAL_UKERNEL=ON

If the path is missing inside the container, clone your oneDNN fork there.

Running Tests

Quick sanity test:

pytest -q tests/test_grouped_gemm_w4a16_onednn.py
Running vLLM with oneDNN Grouped GEMM (MoE int4)

Example:

source /opt/intel/oneapi/setvars.sh

ONEDNN_VERBOSE=all \
VLLM_XPU_GROUPED_GEMM_BACKEND=onednn \
VLLM_LOGGING_LEVEL=DEBUG \
HF_HOME=/data/model \
vllm serve ./gpt-oss-20b-mxfp-w4g256kv \
  --max-model-len 4096 \
  --dtype=float16 \
  --enforce-eager

Key flag:

VLLM_XPU_GROUPED_GEMM_BACKEND=onednn

Without this, CUTLASS int4 path will throw.

Debugging Knobs (Grouped GEMM / MoE)

Enable detailed oneDNN debug:

export VLLM_XPU_ONEDNN_GROUPED_GEMM_DEBUG=1

Other toggles:

export VLLM_XPU_ONEDNN_DISABLE_BIAS=1
export VLLM_XPU_FUSED_MOE_ACTIVATION_OVERRIDE=silu

Clear torch.compile artifacts:

rm -rf /tmp/torchinductor_${USER}
Known Issues
1) libccl.so.2 ImportError

Almost always fixed by:

source /opt/intel/oneapi/setvars.sh
2) CUTLASS int4 grouped GEMM stub

If you see:

"cutlass kernel was called"

You either:

Forgot VLLM_XPU_GROUPED_GEMM_BACKEND=onednn

Are running non-int4 case

3) torch.compile multi-token divergence

Status:

Eager mode works

torch.compile can produce incorrect multi-token output

Likely in attention custom op, not oneDNN GEMM

Workaround:

--enforce-eager
Rebuild Rules
Python-only changes

Effective immediately in editable mode.

Any change under csrc/

Requires rebuild:

pip install --no-build-isolation -e . -v

If CMake is corrupted:

rm -rf build/temp
pip install --no-build-isolation -e . -v
Minimal Dev Loop (Inside Container)
docker exec -it -w $(pwd) hans-gpt-oss-ww17-fresh /bin/bash
source /opt/intel/oneapi/setvars.sh
source .venv/bin/activate
pip install --no-build-isolation -e . -v
pytest -q
