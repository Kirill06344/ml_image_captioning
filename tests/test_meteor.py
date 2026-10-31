import pytest

from src.meteor import CocoMeteor

REFS = {1: ['a cat sits on a mat'], 2: ['a blue dog runs on grass']}
HYPS = {1: ['a cat sits'], 2: ['a dog runs']}


def test_meteor_is_identical_under_russian_and_c_locales(monkeypatch):
    results = []
    for locale in ['C.UTF-8', 'ru_RU.UTF-8']:
        monkeypatch.setenv('LANG', locale)
        monkeypatch.setenv('LC_ALL', locale)
        scorer = CocoMeteor()
        try:
            results.append(scorer.compute_score(REFS, HYPS))
            process = scorer.meteor_p
        finally:
            scorer.close()
        assert process.poll() is not None
    assert results[0][0] == pytest.approx(0.2763378210046511)
    assert results[0] == results[1]


def test_meteor_java_failure_is_reported_without_locked_cleanup(monkeypatch):
    from pycocoevalcap.meteor.meteor import Meteor
    def fail(self, gts, res):
        self.lock.acquire()
        self._errors.write(b'java.lang.TestFailure: diagnostic')
        raise ValueError('empty output')
    monkeypatch.setattr(Meteor, 'compute_score', fail)
    scorer = CocoMeteor()
    with pytest.raises(RuntimeError, match='java.lang.TestFailure: diagnostic'):
        scorer.compute_score(REFS, HYPS)
    assert not scorer.lock.locked()
    assert scorer.meteor_p is None
    scorer.close()
