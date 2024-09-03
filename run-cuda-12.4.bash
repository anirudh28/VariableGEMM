#!/bin/bash

args=
for i in "$@"; do
  i="${i//\\/\\\\}"
  args="${args} \"${i//\"/\\\"}\""
done

if [ "${args}" == "" ]; then args="/bin/bash"; fi

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

export SINGULARITY_BINDPATH=/home,/scratch

singularity exec --bind /usr/local/cuda/bin/cuda-memcheck,/usr/bin/vim,/misc/linux/centos7/x86_64/local/stow/cmake-3.22.0-linux-x86_64/share/cmake-3.22,/usr/local/stow/cmake-3.22.0-linux-x86_64/bin/cmake:/misc/linux/centos7/x86_64/local/stow/cmake-3.22.0-linux-x86_64/bin/cmake ${nv} \
/tmp/cuda-12.4.sif \
/bin/bash -c "
alias cmake=/misc/linux/centos7/x86_64/local/stow/cmake-3.22.0-linux-x86_64/bin/cmake
unset -f which
${args}
"