create folder datasets/FreeBase-dump

download data into datasets/FreeBase-dump

Restart based on `ref_KG_projects/ToG`

Create python env

Install virtuoso 7 opensource with apt

sudo systemctl restart virtuoso-opensource-7

change `virtuoso.ini` to allow read from datasets/FreeBase-dump



stop in the end, sudo systemctl stop virtuoso-opensource-7