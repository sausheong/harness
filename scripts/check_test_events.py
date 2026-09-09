"""Reject skipped, failed, empty or unfinished Go qualification streams."""
import json
from pathlib import Path
import sys


def check(raw):
    active=set(); packages=set();passed_packages=set();tests=0;errors=[]
    for number,line in enumerate(raw.splitlines(),1):
        try:
            event=json.loads(line)
            if not isinstance(event,dict):raise ValueError('object required')
        except (ValueError,TypeError):
            errors.append(f'line {number}: invalid JSON event');continue
        action=event.get('Action');package=event.get('Package');test=event.get('Test')
        if action in ('output','build-output'):continue
        if action=='build-fail':errors.append('build failure');continue
        if not isinstance(package,str) or not package:
            errors.append(f'line {number}: package missing');continue
        if not test:
            if action=='start':
                if package in packages:errors.append('duplicate package start: '+package)
                packages.add(package)
            elif action=='pass':
                if package not in packages or package in passed_packages:errors.append('unmatched package pass: '+package)
                passed_packages.add(package)
            elif action in ('fail','skip'):errors.append(action+' package: '+package)
            else:errors.append('unknown package event: '+str(action))
            continue
        key=(package,test)
        if package not in packages or package in passed_packages:errors.append('test outside active package: '+str(key))
        if action=='run':
            if key in active:errors.append('duplicate running test: '+str(key))
            active.add(key)
        elif action in ('pass','fail','skip'):
            if key not in active:errors.append('unmatched test result: '+str(key))
            active.discard(key)
            if action!='pass':errors.append(action+' test: '+str(key))
            else:tests+=1
        elif action not in ('pause','cont'):errors.append('unknown test event: '+str(action))
    if not tests:errors.append('no passing test records')
    if not packages or packages!=passed_packages:errors.append('missing successful package completion')
    if active:errors.append('unfinished tests: '+str(sorted(active)))
    return dict(status='failed' if errors else 'passed',passed_test_records=tests,packages=len(packages),errors=errors)


if __name__=='__main__':
    result=check(Path(sys.argv[1]).read_text());print(json.dumps(result,indent=2));sys.exit(result['status']!='passed')
