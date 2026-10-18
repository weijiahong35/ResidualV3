"""Persist successful tuning and stop retrying a batch size that killed the server."""


def select_concurrency(results, rejected=()):
    stable = [r for r in results if r['concurrency'] not in rejected
              and r.get('score_tokens_per_second', 0) > 0]
    if not stable:
        return 40
    best = max(r['score_tokens_per_second'] for r in stable)
    return min(r['concurrency'] for r in stable
               if r['score_tokens_per_second'] >= best * .97)


def tuning_candidates(tuning):
    # A finalized selection must survive a restart, even when a failed candidate
    # prevented the original measurement list from reaching five entries.
    if tuning.get('selected_concurrency') is not None:
        return []
    tested = {r['concurrency'] for r in tuning.get('results', [])}
    rejected = {r['concurrency'] for r in tuning.get('failures', [])}
    return [n for n in [40, 64, 80, 96, 128] if n not in tested | rejected]


def record_server_failure(tuning, concurrency, error):
    updated = {**tuning, 'results': list(tuning.get('results', [])),
               'failures': [*tuning.get('failures', []),
                            {'concurrency': concurrency, 'error': str(error)}]}
    rejected = {r['concurrency'] for r in updated['failures']}
    # Freeze at the measured stable choice instead of proceeding to a larger
    # untested batch on the next supervisor start.
    updated['selected_concurrency'] = select_concurrency(updated['results'], rejected)
    if concurrency <= updated['selected_concurrency']:
        updated['selected_concurrency'] = max(1, min(40, concurrency // 2))
    updated['selection'] = 'Server failure: persist a lower stable cap; automatic tuning stopped'
    return updated
