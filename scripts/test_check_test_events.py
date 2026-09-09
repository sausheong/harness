import json
import unittest
from check_test_events import check


def stream(*events):
    return '\n'.join(json.dumps(dict(Package='p',**e)) for e in events)


class EventGateTests(unittest.TestCase):
    def valid(self):
        return [dict(Action='start'),dict(Action='run',Test='TestOne'),dict(Action='pass',Test='TestOne'),dict(Action='pass')]

    def test_complete_and_repeated_tests(self):
        e=self.valid();self.assertEqual(check(stream(*e))['status'],'passed')
        e[3:3]=e[1:3];self.assertEqual(check(stream(*e))['passed_test_records'],2)

    def test_skipped_or_failed_is_never_green(self):
        for action in ('skip','fail'):
            e=self.valid();e[2]['Action']=action;self.assertEqual(check(stream(*e))['status'],'failed')

    def test_missing_and_unmatched_events(self):
        e=self.valid()
        for i in range(len(e)):
            self.assertEqual(check(stream(*(e[:i]+e[i+1:])))['status'],'failed')

    def test_empty_and_malformed(self):
        for raw in ('','not json','[]',stream(dict(Action='start'),dict(Action='pass'))):
            self.assertEqual(check(raw)['status'],'failed')

    def test_hidden_build_failure(self):
        raw=stream(*self.valid())+'\n'+json.dumps(dict(Action='build-fail',ImportPath='other'))
        self.assertEqual(check(raw)['status'],'failed')


if __name__=='__main__':unittest.main()
