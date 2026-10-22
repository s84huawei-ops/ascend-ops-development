#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
for setup in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/cann-9.0.1/set_env.sh; do
    if [[ -f $setup ]]; then source "$setup"; break; fi
done
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.1}
export PYTHONUNBUFFERED=1
# Docker's single device mount already maps the selected physical NPU to logical 0.
# A physical ID in ASCEND_RT_VISIBLE_DEVICES would mask that logical device out.
unset ASCEND_RT_VISIBLE_DEVICES
exec "$@"
