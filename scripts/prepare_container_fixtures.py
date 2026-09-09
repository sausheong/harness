"""Prepare explicit local Docker fixtures and record immutable test settings.

Images must already be loaded. This script never pulls images or starts a daemon.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def prepare(docker, socket, image, bash_image, output):
    for path in (docker, socket, output):
        if not Path(path).is_absolute():
            raise ValueError('absolute Docker, socket and output paths required')
    if any(c in socket for c in ('\n', '\r', '\0')):
        raise ValueError('socket path cannot contain environment delimiters')
    for value in (image, bash_image):
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', value):
            raise ValueError('loaded immutable image IDs required')
    output=Path(output);output.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(prefix='harness-fixtures-') as temporary:
        root=Path(temporary);config=root/'config';config.mkdir()
        prefix=[docker,'--config',str(config),'--host','unix://'+socket]
        def run(args):
            return subprocess.check_output(prefix+args,stderr=subprocess.PIPE,timeout=120)
        inspected=[]
        for identity in (image,bash_image):
            info=json.loads(run(['image','inspect',identity]))[0]
            if info['Id']!=identity or info['Os']!='linux':
                raise ValueError('Linux image identity mismatch')
            inspected.append(dict(id=identity,architecture=info['Architecture']))
        if inspected[0]['architecture']!=inspected[1]['architecture']:
            raise ValueError('fixture image architectures differ')
        dockerfile=('FROM '+image+'\nVOLUME /implicit-fixture-volume\n').encode()
        (root/'Dockerfile').write_bytes(dockerfile)
        # Untagged derived image is identified solely by its captured ID.
        raw=run(['build','--pull=false','--network=none','--quiet',str(root)])
        volume=raw.decode().strip()
        if not re.fullmatch(r'sha256:[0-9a-f]{64}',volume):
            raise ValueError('derived image ID missing')
        info=json.loads(run(['image','inspect',volume]))[0]
        if '/implicit-fixture-volume' not in (info['Config'].get('Volumes') or {}):
            raise ValueError('negative volume fixture missing implicit volume')
        settings=dict(HARNESS_TEST_CONTAINER_IMAGE=image,HARNESS_TEST_BASH_IMAGE=bash_image,
                      HARNESS_TEST_VOLUME_IMAGE=volume,HARNESS_TEST_CONTAINER_SOCKET=socket)
        report=dict(status='prepared',environment=settings,images=inspected,
                    volume_dockerfile_sha256=hashlib.sha256(dockerfile).hexdigest(),
                    note='Loaded images are retained; no tests or publication performed.')
        (output/'fixtures.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'fixtures.env').write_text(''.join(k+'='+v+'\n' for k,v in settings.items()))
        return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('docker','socket','image','bash-image','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=prepare(a.docker,a.socket,a.image,a.bash_image,a.output);print(json.dumps(r))
