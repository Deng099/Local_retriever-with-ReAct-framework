"""Small configuration guard for resuming one experiment directory."""
import hashlib
import json


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ensure_experiment(output_dir, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'experiment.json'
    experiment_id = fingerprint(config)
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing.get('experiment_id') != experiment_id or existing.get('config') != config:
            raise ValueError('Experiment configuration changed; choose a new --output-dir')
    else:
        if any(output_dir.glob('run_*.json')):
            raise ValueError('Legacy results have no experiment identity; choose a new --output-dir')
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'experiment_id': experiment_id, 'config': config},
                                        ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    return experiment_id
