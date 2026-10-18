"""Bounded retries for read-only local hidden requests, never optimizer work."""
import hashlib
import json
from pathlib import Path
from threading import Lock
import time

import openai


def request_with_retries(request, *, log_path, token_ids, max_retries=4, pause=time.sleep):
    identity = hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()
    for attempt in range(max_retries + 1):
        try:
            return request()
        except openai.APIConnectionError as error:
            record = dict(event='hidden_connection_retry', attempt=attempt + 1,
                          total_attempts=max_retries + 1, token_ids_sha256=identity,
                          error_class=type(error).__name__, exhausted=attempt == max_retries,
                          unix_time=time.time())
            with _LOG_LOCK:
                with Path(log_path).open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
            print('HIDDEN_CONNECTION_RETRY', record, flush=True)
            if attempt == max_retries:
                raise
            pause(min(2 ** attempt, 8))


_LOG_LOCK = Lock()


def install(log_path):
    import httpx2
    import runtime.vllm_hidden as hidden
    import speculators.data_generation.vllm_client as client

    original_provider = hidden.OnlineHiddenProvider
    original_request = client.generate_hidden_states.__wrapped__

    class ResilientHiddenProvider(original_provider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            endpoint = str(self.client.base_url)
            self.client.close()
            # Fresh local connections avoid a server/client keep-alive race.
            # The explicit retry above is the only retry layer.
            http = openai.DefaultHttpxClient(
                limits=httpx2.Limits(max_connections=24, max_keepalive_connections=0),
                trust_env=False)
            self.client = openai.OpenAI(base_url=endpoint, api_key='EMPTY',
                                        max_retries=0, http_client=http)

    def generate_hidden_states(client_instance, model, client_item, *, timeout=180, max_retries=0):
        assert max_retries == 0, 'Only the recovery transport may retry'
        return request_with_retries(
            lambda: original_request(client_instance, model, client_item, timeout=timeout),
            log_path=log_path, token_ids=client_item['input_ids'])

    hidden.OnlineHiddenProvider = ResilientHiddenProvider
    client.generate_hidden_states = generate_hidden_states
