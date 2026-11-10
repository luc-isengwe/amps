"""amps - Agent Manifest -> Policy Synthesis.

From a declarative security manifest it:

1. builds the attack-path graph of a multi-agent system   (graph.py)
2. finds integrity / confidentiality attack paths          (verify.py)
3. synthesizes a minimal-cost enforcement policy           (synthesis.py)
4. enforces it at run time (tools, memory, messages)       (runtime.py)
5. analyses the impact of configuration changes and
   generates regression tests                              (impact.py, testgen.py)

The package name is provisional and can be renamed.
"""

__version__ = "0.6.0"
