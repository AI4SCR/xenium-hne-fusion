from dotenv import load_dotenv

assert load_dotenv()

import wandb
api = wandb.Api(api_key='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')

# run is specified by <entity>/<project>/<run_id>

run_ids = ['sai19gbw', '4e5375fe', 'obc2l98u', '42qbc5d7']
for run_id in run_ids:
    run = api.run(f"chuv/xe-hne-fus-cell-v0/{run_id}")
    print(f"\n{'='*60}")
    print(f"Run: {run_id}  state={run.state}  name={run.name}")
    print(f"Config: {dict(run.config)}")

    # System logs often contain the traceback
    files = {f.name: f for f in run.files()}
    log_file = files.get('output.log') or files.get('logs/debug.log')
    if log_file:
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            log_file.download(root=tmp, replace=True)
            log_path = os.path.join(tmp, log_file.name)
            with open(log_path) as fh:
                lines = fh.readlines()
        tail = lines[-80:] if len(lines) > 80 else lines
        print(f"\n--- Last log lines ({log_file.name}) ---")
        print("".join(tail))
    else:
        print(f"  Available files: {list(files.keys())}")