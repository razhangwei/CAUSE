# CAUSE: **C**ausality from **A**ttrib**U**tions on **S**equence of **E**vents

## How to Run

1. Install dependency:

    ```shell
    conda env create -n <myenv> -f environment.yml
    conda activate <myenv>
    ```

2. Run the scripts for individual datasets
```
./scripts/run_excitation.sh all
./scripts/run_inhibition.sh all
./scripts/run_syngergy.sh all
./scripts/run_iptv.sh all
./scripts/run_memetracker.sh all
```

If you find this repo useful, please consider to cite:
```
@inproceedings{zhang2020cause,
  title={Cause: Learning granger causality from event sequences using attribution methods},
  author={Zhang, Wei and Panum, Thomas and Jha, Somesh and Chalasani, Prasad and Page, David},
  booktitle={International Conference on Machine Learning},
  pages={11235--11245},
  year={2020},
  organization={PMLR}
}
```