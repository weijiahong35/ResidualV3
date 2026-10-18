"""Use NVML physical IDs for ownership checks, UUID visibility for CUDA execution."""
import subprocess


def assignment(physical_gpu):
    fields=subprocess.check_output(['nvidia-smi','--id',str(physical_gpu),
        '--query-gpu=uuid,pci.bus_id,name','--format=csv,noheader'],text=True).strip().split(',')
    if len(fields)!=3 or not fields[0].strip().startswith('GPU-'):raise RuntimeError('invalid GPU identity')
    return dict(physical_gpu=physical_gpu,uuid=fields[0].strip(),pci_bus_id=fields[1].strip(),name=fields[2].strip(),cuda_device=0)


def bind(env,identity):
    env.update(CUDA_VISIBLE_DEVICES=identity['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
               REPAIR_PHYSICAL_GPU=str(identity['physical_gpu']),REPAIR_GPU_UUID=identity['uuid'])
    return env
