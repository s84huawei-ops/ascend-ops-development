# Ascend operator development framework

Terms used when planning operator versioning, measurements, and results.

## Language

**Operator**:
A computational operation under development, together with its custom Ascend implementation. The demo operator is the first simple example used to validate the framework.

**Operator version**:
An unambiguously identified state of the operator sources used in an experiment.

**Benchmark**:
A specified set of operator invocations used to measure execution time.

**Experiment**:
One execution of a defined configuration, together with the versions used, logs, and results.
_Avoid_: benchmark as the name for the entire experiment

**NPU profiling**:
Collecting information about operator execution on a physical Ascend device.

**Simulator profiling**:
Collecting information about simulated execution of the operator kernel. Results describe a device model and do not measure time on a physical NPU.
