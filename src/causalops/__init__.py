"""CausalOps package entry point.

`TECHNICAL_SPEC.md` section 11 permits `langsmith` to be present only as an
inert transitive dependency of `langchain-core`, "provided tracing is
force-disabled at the entry point." This is that entry point: importing
`causalops` at all -- not just running `causalops.cli:main` -- forces both
tracing variables off.
"""

import os

# Force these off unconditionally (not os.environ.setdefault) so an ambient
# shell value cannot enable tracing for a CausalOps run. LangSmith tracing is
# prohibited outright by TECHNICAL_SPEC.md section 11, not merely off by
# default, so an explicit opt-in from outside this process must not win.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
