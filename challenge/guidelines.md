# Challenge Participation Guidelines (WIP)

This document walks you through the end-to-end workflow: from pulling the base image, developing your model, to submitting and seeing your results on the leaderboard.


## Environment Setup (placeholder)

### Activate your virtual environment
```bash
source .venv/{environment_name}/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
```
### Pull our base Docker image
```bash
docker pull internauto/iros2025-base:latest
```
### Download the starter dataset (val_seen + val_unseen splits)
```bash
wget https://datasets.internrobotics.org/iros2025/starter_data.tar.gz
tar -xf starter_data.tar.gz -C ./data
```
## Local Development & Testing (placeholder)
### Run the container
```bash
docker run -it --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  internauto/iros2025-base:latest \
  /bin/bash
```
### Develop & test
- Implement your policy under `internmanip/your_policy.py`.
- Use our evaluation script for quick checks:
    ```bash
    python scripts/eval/local_eval.py \
      --policy internmanip.your_policy.YourPolicy \
      --data-splits val_seen,val_unseen
    ```
- Placeholder for example code.

## Packaging & Submission (placeholder)
### Copy your trained weights & configs into the image
```bash
# inside container
cp outputs/checkpoint.pt /workspace/models/
```
### Build your submission image
```bash
docker build -t registry.cn-hangzhou.aliyuncs.com/yourteam/iros2025:dev .
```
### Push to the registry
```bash
docker push registry.cn-hangzhou.aliyuncs.com/yourteam/iros2025:dev
```
### Submit your image URL on Eval.AI

#### Submission Format

Create a JSON file with your Docker image URL and team information. The submission must follow this exact structure:

```json
{
    "url": "registry.cn-hangzhou.aliyuncs.com/yourteam/iros2025:dev",
    "team": {
        "name": "your-team-name",
        "members": [
            {
                "name": "John Doe",
                "affiliation": "University of Example",
                "email": "john.doe@example.com",
                "leader": true
            },
            {
                "name": "Jane Smith",
                "affiliation": "Example Research Lab",
                "email": "jane.smith@example.com",
                "leader": false
            }
        ]
    }
}
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Complete Docker registry URL for your submission image |
| `team.name` | string | Official team name for leaderboard display |
| `team.members` | array | List of all team members with their details |
| `members[].name` | string | Full name of team member |
| `members[].affiliation` | string | University or organization affiliation |
| `members[].email` | string | Valid contact email address |
| `members[].leader` | boolean | Team leader designation (exactly one must be `true`) |


For detailed submission guidelines and troubleshooting, refer to the official Eval.AI platform documentation.

## Official Evaluation Flow
### DSW Creation
- We use the AliCloud API to instantiate a DSW from your image link.
- The system mounts our evaluation config + full dataset (val_seen, val_unseen, test).
### Evaluation Execution
- Via SSH + `screen`, we launch `scripts/eval/run_eval.sh`.
- A polling loop watches for `results.json`.
### Results Collection
- Upon completion, metrics for each split are parsed and pushed to Eval.AI leaderboard.

> 😄 Good luck, and may the best vision-based policy win!
