import numpy as np

from evaluation.grounding_metrics import Metrics; from evaluation.grounding_metrics import coverage_group; from grounding.evidence import fixed_scores; from evaluation.grounding_metrics import local_group; from grounding.evidence import packed_evidence; from grounding.element_features import restrict_candidates


def test_multi_positive_groups_and_fallback_group():
    assert coverage_group([1], [1, 2], [2]) == 'A'
    assert coverage_group([1], [1], [1, 2]) == 'B1'
    assert coverage_group([1], [1, 2], [1, 2]) == 'B2'
    assert coverage_group([], [], [2]) == 'C'
    assert local_group([1, 2], [1], [2]) == 'L'
    assert local_group([1, 2], [1], [1, 2]) == 'L_prime'


def test_restriction_preserves_scores_and_fallback():
    row = dict(candidate_ids=[1, 4], features=[[.1, .3], [.7, .9]],
               base_scores=[-2., -4.], fallback=7)
    restricted = restrict_candidates(row, {4})
    assert restricted == dict(candidate_ids=[4], features=[[.7, .9]], base_scores=[-4.], fallback=7)
    assert restrict_candidates(row, set())['fallback'] == 7
    assert row['candidate_ids'] == [1, 4]


def test_fixed_formula_zero_coefficients_and_missing_windows():
    evidence = [dict(ids=[3, 21], logps=[[-2., -3.], [-5.]],
                     priors=[[-.5, -.5], [-.5]], semantics=[-.2, -.4])]
    packed = packed_evidence(evidence)
    scores = fixed_scores(packed, (1., 1., 1., 2.))[0]
    assert np.allclose(scores, [np.logaddexp(-2.5, -3.5)-.2, -7.9])
    assert np.allclose(fixed_scores(packed, (0., 0., 0., 0.))[0], [np.log(2), 0])


def test_contributions_use_full_task_denominators():
    baseline = [dict(sample_id='a', annotation_id='t1'),
                dict(sample_id='b', annotation_id='t1'),
                dict(sample_id='c', annotation_id='t2')]
    m = Metrics(baseline)
    first, second = {'a': True}, {'c': True}
    assert m.difference(first, second)['delta_pp'] == 25
    assert m.difference(first, second, {'a'})['delta_pp'] == -25
    assert m.difference(first, second, {'c'})['delta_pp'] == 50
