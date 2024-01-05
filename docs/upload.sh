
#!/bin/bash

scriptDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rsync -avzh ${scriptDir}/docsTmpFiles/* rr@10.50.103.16:/home/rr/api_docs/docs/grip_docs