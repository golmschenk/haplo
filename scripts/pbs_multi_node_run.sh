#PBS -l select=2:ncpus=2:model=ivy
#PBS -l walltime=0:30:00
#PBS -j oe
#PBS -W group_list=s2853
#PBS -q devel

source /usr/local/lib/init/global.profile

module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4
source activate haplo_env

module load mpi-hpe/mpt

head_node_hostname=`/bin/hostname -s`

export MPI_SHEPHERD=true
export MPI_DSM_DISTRIBUTE=0

mpiexec -perhost 1 python -m torch.distributed.run \
--nnodes 2 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_hostname \
scripts/example_train_session.py
